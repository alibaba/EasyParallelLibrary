# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Classes for doing parallelism and graph optimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import tf_logging
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow._api.v1 import summary as tf_summary

from epl.cluster import Cluster
from epl.cluster import generate_device_slices
from epl.env import Env
from epl.ir.graph import Graph
from epl.ir.graph import GraphKeys
from epl.ir.tensor import Tensor
from epl.parallel.graph_editor import GraphEditor
from epl.parallel import planner
from epl.parallel.ops import allgather_tensors
from epl.parallel.ops import allreduce_tensors
from epl.parallel.ops import create_simple_communicator
from epl.runtime.zero import zero_v1
from epl.utils import constant, common


class OutputSpec(object):
  """Classes to store attributes of output collection of each merge type."""
  def __init__(self, name, devices):
    self._name = name
    self._devices = devices
    self._comm = create_simple_communicator(name, devices)
    # First is tensors to communicate, second is original tensors.
    self._allgather_tensors = [dict(), dict()]
    self._allreduce_mean_tensors = [dict(), dict()]
    self._allreduce_sum_tensors = [dict(), dict()]

  @property
  def comm(self):
    return self._comm

  def get_tensors(self, comm_type):
    """Return tensor list with comm_type."""
    if comm_type == GraphKeys.GLOBAL_CONCAT_OBJECTS:
      return self._allgather_tensors
    if comm_type == GraphKeys.GLOBAL_MEAN_OBJECTS:
      return self._allreduce_mean_tensors
    if comm_type == GraphKeys.GLOBAL_SUM_OBJECTS:
      return self._allreduce_sum_tensors
    return None

  def add_tensor_or_name(self, replica_idx, tensor_or_name, comm_type):
    """Record tensor or tensor_name with comm_type and replica_idx."""
    tensors_index = 1 if isinstance(tensor_or_name, (Tensor, ops.Tensor)) else 0
    if comm_type == GraphKeys.GLOBAL_CONCAT_OBJECTS:
      if replica_idx in self._allgather_tensors[tensors_index]:
        self._allgather_tensors[tensors_index][replica_idx].append(
            tensor_or_name)
      else:
        self._allgather_tensors[tensors_index][replica_idx] = [tensor_or_name]
    elif comm_type == GraphKeys.GLOBAL_MEAN_OBJECTS:
      if replica_idx in self._allreduce_mean_tensors[tensors_index]:
        self._allreduce_mean_tensors[tensors_index][replica_idx].append(
            tensor_or_name)
      else:
        self._allreduce_mean_tensors[tensors_index][replica_idx] = [tensor_or_name]
    elif comm_type == GraphKeys.GLOBAL_SUM_OBJECTS:
      if replica_idx in self._allreduce_sum_tensors[tensors_index]:
        self._allreduce_sum_tensors[tensors_index][replica_idx].append(
            tensor_or_name)
      else:
        self._allreduce_sum_tensors[tensors_index][replica_idx] = [tensor_or_name]

  def serialize(self):
    return "OutputSpec(name = {}, devices = {}, comm = {}, allgather_tensors={}, allreduce_mean_tensors={}, allreduce_sum_tensors={})".format(
        self._name, self._devices, self._comm, self._allgather_tensors, self._allreduce_mean_tensors, self._allreduce_sum_tensors)

  def __str__(self):
    return self.serialize()

  def __repr__(self):
    return self.serialize()


class Parallel(object):
  """"Classes to transform original graph into parallel graph."""
  def __init__(self):
    self.transformer = GraphEditor()
    self._already_replacement = False

  @property
  def _graph(self):
    """Get epl graph."""
    return Graph.get()

  @classmethod
  def get(cls):
    """Get static parallel."""
    tf_graph = common.get_default_tf_graph()
    parallel_map = Env.get().parallel_map
    if tf_graph not in parallel_map:
      parallel_map[tf_graph] = Parallel()
    return parallel_map[tf_graph]

  def device_replacement(self):
    """Set devices for taskgraphs."""
    if self._already_replacement:
      return
    if not Env.get().cluster.virtual_devices:
      Env.get().cluster.generate_virtual_devices("auto")
    virtual_devices = Env.get().cluster.virtual_devices
    for taskgraph in self._graph.taskgraphs:
      vd = virtual_devices[taskgraph.index] if len(virtual_devices) >= len(self._graph.taskgraphs) else virtual_devices[0]
      taskgraph.set_device(vd)
    if self._graph.is_constructor:
      self.transformer.device_replacement()
      self._already_replacement = True
    else:
      self._graph.reset()
      Env.get().get_or_create_server().join()

  def auto_stages(self, num_stages):
    """When auto parallel is enabled, partition stages in constructor."""
    if self._graph.is_constructor:
      device_per_replicas = [1] * num_stages
      num_replica = Env.get().cluster.worker_num
      device_slices = generate_device_slices(Env.get().cluster, device_per_replicas, num_replica)
      Env.get().cluster = Cluster(layout={'specific': device_slices})

      plan = planner.AutoStageGenerator(num_stages=num_stages)
      stage_ops = plan.search()
      self.transformer.partition_stages(stage_ops)

  def fix_dataset(self):
    # Place dataset related ops on GPU will cause core dump for
    # tensorflow 1.15 even opening soft placement. Here we place
    # all dataset replated ops on CPU.
    cpu_device = Env.get().cluster.current_worker_cpu()
    dataset_related_ops = self._graph.get_dataset_related_ops()
    for op in dataset_related_ops:
      op.set_device(cpu_device)

  def _fix_map_operations_to_taskgraph(self):
    """Map operations of APPLY/SAVE_AND_RESTORE phase to taskgraph."""
    def fetch_taskgraph_index(op, stage_idx):
      """Find taskgraph for operations whose phase in [APPLY, SAVE_AND_RESTORE]
      with consumers, inputs and control_inputs."""
      for tensor in op.outputs:
        consumers = tensor.primitive_obj.consumers()
        for consumer in consumers:
          consumer = self._graph.get_operation_by_name(consumer.name)
          if consumer in self._graph.unclustered_operations:
            continue
          assert consumer.taskgraph, "{} expected to be related to some taskgraph".format(consumer)
          stage_idx = consumer.taskgraph.index if stage_idx == -1 else min(stage_idx, consumer.taskgraph.index)
      for tensor in list(op.inputs):
        tensor = self._graph.get_tensor_by_name(tensor.name)
        if tensor.taskgraph is not None:
          stage_idx = max(tensor.taskgraph.index, stage_idx)
          continue
      for op_or_tensor in op.control_inputs:
        assert isinstance(op_or_tensor, ops.Operation), "Control input expected type of tf.Operation, while {}".format(op_or_tensor)
        c_inp_op = self._graph.get_operation_by_name(op_or_tensor.name)
        if c_inp_op.taskgraph:
          stage_idx = max(c_inp_op.taskgraph.index, stage_idx)
      return stage_idx

    remainder = len(self._graph.unclustered_operations)
    while self._graph.unclustered_operations:
      unclustered_operations = list(self._graph.unclustered_operations)
      for op in unclustered_operations:
        stage_idx = fetch_taskgraph_index(op, -1)
        if stage_idx != -1:
          self._graph.unclustered_operations.remove(op)
          taskgraph = self._graph.taskgraphs[stage_idx]
          self._graph.link_operation_to_taskgraph(op, taskgraph)
      if len(self._graph.unclustered_operations) == remainder:
        break
      remainder = len(self._graph.unclustered_operations)
    if self._graph.unclustered_operations:
      # Postprecess operations without consumers, inputs and control_inputs,
      # these operations act as control_input of some operation and
      # act as init op of tensorflow graph.
      for op in list(self._graph.unclustered_operations):
        for c_inp_of_others in op.control_inputs_consumers:
          if c_inp_of_others in self._graph.unclustered_operations:
            continue
          assert c_inp_of_others.taskgraph, "{} expected to be related to some taskgraph".format(c_inp_of_others)
          stage_idx = c_inp_of_others.taskgraph.index if stage_idx == -1 else min(stage_idx, c_inp_of_others.taskgraph.index)
        self._graph.unclustered_operations.remove(op)
        taskgraph = self._graph.taskgraphs[stage_idx]
        self._graph.link_operation_to_taskgraph(op, taskgraph)
      assert not self._graph.unclustered_operations, \
          "Expect no operations left without taskgraph info, but {}".format(self._graph.unclustered_operations)

  def do_parallelism(self):
    """Tranform original graph to parallel graph."""
    if self._graph.is_constructor:
      with ops.name_scope(constant.PARALLEL_STRATEGY):
        for op in self._graph.dataset_api_op:
          if op.type in constant.ODPS_TABLE_API_OPS:
            self._graph.check_and_set_cloned_dataset_need_clone()
        self._fix_map_operations_to_taskgraph()
        if Env.get().config.offload.level == "v0":
          tf_logging.info("enable weight offload")
          self.transformer.offload_weight()
        self.transformer.micro_batch_clone()
        self.transformer.replicas_clone()
        if not zero_v1():
          # if zero_level is v1, then use reduce instead of allreduce.
          self.transformer.gradient_aggregation()
        self.transformer.schedule_optimization()
        if self._graph.num_constructors >= 1 and Env.get().config.io.slicing:
          self.transformer.io_slicing()
        self.merge_outputs()
        self.update_summaries()

  def merge_outputs(self):
    """Merge outputs according to graph.collections."""
    def is_defined_in_other_collection(tensor, current_key):
      """Check if some tensor in other collections."""
      redefined_flag = False
      for graph_key in GraphKeys.ALL_COLLECTION_KEYS:
        if graph_key == current_key:
          continue
        if tensor in self._graph.get_collection(graph_key):
          self._graph.get_collection(graph_key).remove(tensor)
          redefined_flag = True
      return redefined_flag

    # Map devices to communicator
    output_specs = dict()
    communicator_count = 0
    communicator_prefix = "Comm_"
    for graph_key in GraphKeys.ALL_COLLECTION_KEYS:
      tensor_list = list(self._graph.get_collection(graph_key))
      for tensor in tensor_list:
        if is_defined_in_other_collection(tensor, graph_key):
          self._graph.get_collection(graph_key).remove(tensor)
          tf_logging.warn(
              "Tensor {} re-defined in multi-collections.")
          continue
        original_tensor = tensor
        taskgraph = tensor.taskgraph
        if taskgraph.strategy_context.split_strategy:
          continue
        local_micro_batches = self._graph.get_local_micro_batches(tensor)
        local_replicas = self._graph.get_local_replicas(tensor)
        # Merge outputs locally if pipeline enabled
        # and tensor has micro-batches.
        if self._graph.pipeline_enabled and local_micro_batches:
          local_micro_batches = [micro_batch.primitive_obj for micro_batch in local_micro_batches]
          local_micro_batches.append(tensor.primitive_obj)
          if graph_key in [GraphKeys.LOCAL_CONCAT_OBJECTS, GraphKeys.GLOBAL_CONCAT_OBJECTS]:
            local_merged_fn = array_ops.concat
            axis = 0
          elif graph_key in [GraphKeys.LOCAL_SUM_OBJECTS, GraphKeys.GLOBAL_SUM_OBJECTS]:
            local_merged_fn = math_ops.reduce_sum
            axis = None
          elif graph_key in [GraphKeys.LOCAL_MEAN_OBJECTS, GraphKeys.GLOBAL_MEAN_OBJECTS]:
            local_merged_fn = math_ops.reduce_mean
            axis = None
          else:
            local_merged_fn = None
          if local_merged_fn is not None:
            with ops.device(tensor.device):
              tensor = local_merged_fn(local_micro_batches, axis)
              # Record merged_tensor for tensor in original replica
              # when only merged locally.
              self._graph.merged_outputs_map[original_tensor.name] = tensor
        # Merge outputs cross constructors.
        if taskgraph.num_replicas > 1:
          all_devices = taskgraph.virtual_device.all_devices
          device_str = ",".join(all_devices)
          if graph_key in [GraphKeys.GLOBAL_CONCAT_OBJECTS, GraphKeys.GLOBAL_SUM_OBJECTS, GraphKeys.GLOBAL_MEAN_OBJECTS]:
            if device_str not in output_specs:
              output_specs[device_str] = OutputSpec(communicator_prefix + str(communicator_count), all_devices)
              communicator_count += 1
            output_specs[device_str].add_tensor_or_name(0, tensor, graph_key)
            output_specs[device_str].add_tensor_or_name(0, original_tensor.name, graph_key)

          replicated_name = original_tensor.name + constant.MERGED_REPLICAS_SUFFIX
          for replica_idx in range(1, taskgraph.local_num_replicas):
            if not local_replicas:
              break
            replicated_tensor = local_replicas[replica_idx - 1]
            replicated_local_micro_batches = self._graph.get_local_micro_batches(self._graph.get_tensor_by_name(replicated_tensor.name))
            # Merge outputs locally if pipeline enabled
            # and replicated_tensor has micro-batches.
            if self._graph.pipeline_enabled and replicated_local_micro_batches and local_merged_fn is not None:
              replicated_local_micro_batches = [micro_batch.primitive_obj for micro_batch in replicated_local_micro_batches]
              replicated_local_micro_batches.append(
                  replicated_tensor.primitive_obj)
              with ops.device(replicated_tensor.device):
                replicated_tensor = local_merged_fn(replicated_local_micro_batches, axis)
            if device_str in output_specs:
              output_specs[device_str].add_tensor_or_name(
                  replica_idx, replicated_tensor, graph_key)
              output_specs[device_str].add_tensor_or_name(
                  replica_idx, replicated_name, graph_key)

    def update_merged_outputs_map(output_spec, graph_key, merged_fn, mean):
      """Update merged outputs map of tensors in some collection."""
      comm_tensors = output_spec.get_tensors(graph_key)
      control_dependency_list = list()
      for replica_idx, tensor_list in list(comm_tensors[1].items()):
        tensor_list = merged_fn(output_spec.comm,
                                [tensor.primitive_obj if isinstance(tensor, Tensor) else tensor for tensor in tensor_list],
                                tensor_list[0].device,
                                mean)
        for tensor_idx, tensor in enumerate(tensor_list):
          original_tensor_name = comm_tensors[0][replica_idx][tensor_idx]
          if replica_idx:
            if original_tensor_name in self._graph.merged_outputs_map:
              self._graph.merged_outputs_map[original_tensor_name].append(
                  tensor)
            else:
              self._graph.merged_outputs_map[original_tensor_name] = [tensor]
          else:
            self._graph.merged_outputs_map[original_tensor_name] = tensor
          control_dependency_list.append(tensor)
      return control_dependency_list

    comm_graph_keys = [GraphKeys.GLOBAL_CONCAT_OBJECTS, GraphKeys.GLOBAL_MEAN_OBJECTS, GraphKeys.GLOBAL_SUM_OBJECTS]
    for _, output_spec in list(output_specs.items()):
      control_dependency_list = []
      for graph_key in comm_graph_keys:
        if graph_key == GraphKeys.GLOBAL_CONCAT_OBJECTS:
          merged_fn = allgather_tensors
          mean = None
        elif graph_key == GraphKeys.GLOBAL_MEAN_OBJECTS:
          merged_fn = allreduce_tensors
          mean = True
        elif graph_key == GraphKeys.GLOBAL_SUM_OBJECTS:
          merged_fn = allreduce_tensors
          mean = False
        with ops.control_dependencies(control_dependency_list):
          control_dependency_list = update_merged_outputs_map(output_spec, graph_key, merged_fn, mean)

  def update_summaries(self):
    """Update summaries with outputs merged."""
    def fetch_summary_fn(summary_type):
      """Return relative summary function."""
      if summary_type == constant.SUMMARY_SCALAR_TYPE:
        return tf_summary.scalar
      if summary_type == constant.SUMMARY_IMAGE_TYPE:
        return tf_summary.image
      if summary_type == constant.SUMMARY_HISTOGRAM_TYPE:
        return tf_summary.histogram
      if summary_type == constant.SUMMARY_AUDIO_TYPE:
        return tf_summary.audio
      if summary_type == constant.SUMMARY_TEXT_TYPE:
        return tf_summary.text
      if summary_type == constant.SUMMARY_TENSOR_TYPE:
        return tf_summary.tensor
      tf_logging.warn("Unsupported summary type to merge for {}".format(summary_type))
      return None

    def update_summary_inputs(consumer, in_idx, tensor):
      """Update inputs of summary with outputs merged."""
      taskgraph = consumer.taskgraph
      for replica_idx in range(taskgraph.local_num_replicas):
        replica_prefix = common.get_replica_prefix(replica_idx)
        for micro_batch_idx in range(taskgraph.pipeline_config.num_micro_batch):
          micro_batch_prefix = common.get_micro_batch_prefix(micro_batch_idx)
          consumer_name = replica_prefix + micro_batch_prefix + consumer.name
          if consumer_name not in self._graph.operations:
            continue
          consumer_to_be_update = self._graph.get_operation_by_name(consumer_name)
          consumer_to_be_update.update_input(in_idx, tensor)

    summary_key = ops.GraphKeys.SUMMARIES
    summaries = ops.get_collection_ref(summary_key)
    summaries_list = list(summaries)
    for summary in summaries_list:
      if not isinstance(summary, ops.Tensor):
        continue
      if summary.name not in self._graph.summary_map:
        continue
      if self._graph.summary_map[summary.name].tensor_name not in self._graph.merged_outputs_map:
        continue
      summary_fn = fetch_summary_fn(self._graph.summary_map[summary.name].summary_type)
      if not summary_fn:
        continue
      summaries.remove(summary)
      with ops.device(summary.device):
        val = summary_fn(self._graph.summary_map[summary.name].tags,
                         self._graph.merged_outputs_map[self._graph.summary_map[summary.name].tensor_name])
      self._graph.summary_map.pop(summary.name)
      consumers = self._graph.get_tensor_by_name(summary.name).consumers
      for consumer in consumers:
        for in_idx, inp in enumerate(consumer.primitive_obj.inputs):
          if inp.name != summary.name:
            continue
          for tensor in ops.get_collection(summary_key):
            if tensor.name == val.name:
              update_summary_inputs(consumer, in_idx, tensor)
              break

  def graph_optimize(self):
    pass

  def saver_refine(self):
    pass
