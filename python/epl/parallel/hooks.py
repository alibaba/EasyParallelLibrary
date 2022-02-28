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
"""Hooks for epl do parallelism in appropriate time."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os
import warnings
import types
from collections import Counter
from collections import OrderedDict
from distutils.version import LooseVersion as Version
import six

import tensorflow
from tensorflow._api.v1 import layers
from tensorflow._api.v1 import linalg
from tensorflow._api.v1 import losses
from tensorflow._api.v1 import math
from tensorflow._api.v1 import summary
from tensorflow.core.protobuf import config_pb2
from tensorflow.keras.layers import Layer
from tensorflow.python.client import session
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import run_config
from tensorflow.python.estimator import training
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.framework.versions import __version__
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.variables import RefVariable
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import monitored_session
from tensorflow.python.training import optimizer
from tensorflow.python.training import saver

from epl.communicators.collective_communicator import estimate_split_num_for_comm
from epl.env import Env
from epl.ir.graph import Graph
from epl.ir.phase import ModelPhase
from epl.ops import distributed_ops
from epl.ops import initializers
from epl.ops.distributed_dense import distributed_dense
from epl.ops.distributed_losses import distributed_sparse_softmax_cross_entropy_with_logits
from epl.parallel.ops import dispatch_across_consumers
from epl.parallel.ops import alltoall
from epl.parallel.ops import create_serial_communicator
from epl.parallel.ops import create_simple_communicator
from epl.parallel.parallel import Parallel
from epl.runtime.gc import gradient_checkpoint
from epl.runtime.amp.auto_mixed_precision import AMP, amp_enabled
from epl.runtime.amp.loss_scale import amp_loss_scale, amp_update, scale_loss, unscale_grads
from epl.runtime.gradient_accumulation import apply_ga
from epl.runtime.gradient_accumulation import ga_enabled, ga_iter_num
from epl.runtime.optimizer_helper import apply_grad_group
from epl.runtime.zero import zero_enabled, apply_zero
from epl.strategies.replicate import Replicate
from epl.utils import common
from epl.utils import constant
from epl.utils.common import update_tuple
from epl.utils.summary_info import SummaryInfo


# Ensure add_hooks function is only called once.
IS_EPL_HOOKED = False


def graph_add_operation(fn):
  def add_operation(self, op):
    fn(self, op)
    Graph.get(may_create=True).add_operation(op)

  return add_operation


def control_flow_add_while_context(fn):
  """Hook AddWhileContext to get backward operations for while_loop."""
  def add_while_context(self, op, between_op_list, between_ops):
    Graph.get().set_current_backward_op(op)
    return fn(self, op, between_op_list, between_ops)

  return add_while_context


def gradients_impl_maybe_compile(fn):
  """Hook gradient function to get backward operations."""
  def maybe_compile(self, op, func, grad_fn):
    Graph.get().set_current_backward_op(op)
    return fn(self, op, func, grad_fn)

  return maybe_compile


def gradients_impl_gradients_helper(fn):
  """Check epl context."""
  def gradient_helper(*args, **kwargs):
    """Check is epl context empty before gradients function. Clear
    epl context after gradeints function. Get gradients results."""
    if Env.get().strategy_context:
      warnings.warn("EPL ignores the context of backward operations, "
                    "as it collocates the backward operations with their "
                    "corresponding forward operations automatically.")

    config = Env.get().config
    if config.auto.auto_parallel:
      Graph.get().taskgraphs[0].strategy_context.add_context(Replicate(1))
      num_stages = Env.get().config.pipeline.num_stages
      if num_stages > 1:
        tf_logging.info("Enable auto partition model into {} taskgraphs.".format(num_stages))
        Parallel.get().auto_stages(num_stages)
    if amp_enabled():
      tf_logging.info("Enable AMP with loss scale {}".format(config.amp.loss_scale))
      scale = amp_loss_scale()
      loss = scale_loss(args[0], scale)
      args = (loss,) + args[1:]
    Parallel.get().device_replacement()
    # EPL should place the computation of gradients on the same
    # device as the original (forward-pass) op. Tensorflow of version 1.15 has
    # parameter unconnected_gradients which not exist in tensorflow 1.12. To
    # adapt different versions of tensorflow, use args and kwargs instead of
    # concrete parameters.
    colocate_gradients_with_ops_key_position = 4
    assert len(args) > colocate_gradients_with_ops_key_position + 1, \
      "args len {} should > colocate_gradients_with_ops_key_position + 1: {}" \
      .format(len(args), colocate_gradients_with_ops_key_position + 1)
    args = update_tuple(args, True, colocate_gradients_with_ops_key_position)

    with ModelPhase(ModelPhase.BACKWARD):
      gradient_checkpoints = Env.get().config.gradient_checkpoint.type
      if gradient_checkpoints:
        gradients = gradient_checkpoint.gradients(fn, gradient_checkpoints, *args, **kwargs)
      else:
        gradients = fn(*args, **kwargs)
      with ops.name_scope("gradients"):
        for index, grad in enumerate(gradients):
          # Update input will cause error for shape changed for sparse gradients
          # with determinated shape at dim 0.
          if common.is_indexed_slices(grad):
            if Env.get().config.communication.sparse_as_dense or grad.values.shape[0] is not None:
              gradients[index] = ops.convert_to_tensor(grad)

    if amp_enabled():
      AMP().convert()
      with ModelPhase(ModelPhase.BACKWARD):
        with ops.name_scope(constant.LOSS_SCALE_SCOPE_NAME):
          gradients = unscale_grads(gradients, scale)
    if Env.get().config.communication.clip_after_allreduce:
      if not ga_enabled():
        warnings.warn("Clip gradients by norm after allreduce is enabled.")
        Graph.get().gradients = gradients
      else:
        warnings.warn("Gradient accumulation does not support opt_clip_after_allreduce by now, ignore it.")
    return gradients

  return gradient_helper


def optimizer_apply_gradients(fn):
  """Hook apply function to get apply operations."""
  def apply_gradients(self, grads_and_vars, *args, **kwargs):
    """Get apply operations by set model phase to APPLY."""
    if Env.get().strategy_context:
      warnings.warn("EPL ignores the context of apply operations, "
                    "as it collocates the apply operations with their "
                    "corresponding forward operations automatically.")

    num_apply_group = Env.get().config.optimizer.num_apply_group

    if not Env.get().config.communication.clip_after_allreduce:
      # TODO(jiangle.jl): Suppoort multi-optimizers.
      if not ga_enabled():
        Graph.get().gradients += [gv[0] for gv in grads_and_vars]


    global_step = None
    name = None
    if len(args) >= 2:
      global_step = args[0]
      name = args[1]
    if global_step is None:
      global_step = kwargs.get('global_step')
    if name is None:
      name = kwargs.get('name')

    ga_iters = ga_iter_num()

    with ModelPhase(ModelPhase.APPLY):
      apply_fn = None
      if zero_enabled():
        apply_fn = lambda: apply_zero(self, fn, grads_and_vars,
                                      global_step, ga_iters,
                                      num_apply_group, name)
      elif ga_enabled():
        apply_fn = lambda: apply_ga(self, fn, grads_and_vars,
                                    global_step, ga_iters,
                                    num_apply_group, name)
      elif num_apply_group > 1:
        apply_fn = lambda: apply_grad_group(self, fn, grads_and_vars,
                                            num_apply_group,
                                            global_step, name=name)
      else:
        apply_fn = lambda: fn(self, grads_and_vars, *args, **kwargs)

      if amp_enabled() and Env.get().config.amp.loss_scale == "dynamic":
        return amp_update(grads_and_vars, apply_fn, name)
      return apply_fn()
  return apply_gradients


def graph_init(fn):
  def init(self, *args, **kwargs):
    ret = fn(self, *args, **kwargs)
    if Env.get().cluster:
      Env.get().cluster.set_default_device(self)
    return ret
  return init


def graph_finalize(fn):
  """Hook graph finalize function."""
  def finalize(self):
    """Clear init ops collections to avoid initialzing local resources twice
    and do parallelism."""
    # Local resources has been initialized. To avoid initializing local
    # resources twice, we have to clear init ops collections which
    # stored initialized local resources.
    Env.get().parallel_information[constant.ALL_COMM_RESOURCES].extend(resources.local_resources())
    ops.get_default_graph().clear_collection(ops.GraphKeys.LOCAL_RESOURCES)
    ops.get_default_graph().clear_collection(ops.GraphKeys.LOCAL_INIT_OP)
    ops.get_default_graph().clear_collection(ops.GraphKeys.RESOURCES)

    Parallel.get().device_replacement()
    if Graph.get().need_parallel:
      Parallel.get().do_parallelism()
    # For tensorflow 1.15 dataset.
    Parallel.get().fix_dataset()

    return fn(self)

  return finalize


# pylint: disable=protected-access
def scaffold_finalize(fn):
  def finalize(self):
    fn(self)
    Graph.get().primitive_init_op = self._init_op

  return finalize


def base_session_init(fn):
  """Initialize a server before session init."""
  def init(self, target="", graph=None, config=None):
    """Base session init."""
    # If target is none, init training server.
    if target:
      raise ValueError("Target should be none and you should remove your "
                       "server creation code, because server is already "
                       "created by epl.")
    target = Env.get().get_or_create_server().target

    # Force allow_soft_placement to be true.
    if config is None:
      config = config_pb2.ConfigProto(allow_soft_placement=True)
    else:
      if not isinstance(config, config_pb2.ConfigProto):
        raise TypeError("Config must be a tf.ConfigProto, but got %s." % type(config))
      config.allow_soft_placement = True

    return fn(self, target, graph, config)

  return init


def variable_scope_init(fn):
  def init(self, *args, **kwargs):
    """Init function of variable scope."""
    res = fn(self, *args, **kwargs)
    Graph.get(may_create=True).map_variable_scope_to_taskgraph(self.name)
    return res

  return init


def monitored_session_init(fn):
  """Set SAVE_AND_RESTORE phase for adding a new taskgraph to save
  saver related ops"""
  def init(self, *args, **kwargs):
    """Init function of _MonitoredSession"""
    Graph.get().set_model_phase(ModelPhase.SAVE_AND_RESTORE)
    res = fn(self, *args, **kwargs)
    return res

  return init


def is_apply_related(tensor):
  """Is tensor apply related."""
  return Graph.get().operations[tensor.op.name].phase == ModelPhase.APPLY


def broadcast_variables():
  """Broadcast weight values of trainable variables."""
  graph = Graph.get()
  assign_ops = []
  for taskgraph_idx, taskgraph in enumerate(graph.taskgraphs):
    if taskgraph.num_replicas <= 1 or \
        taskgraph.strategy_context.replicate_strategy is None:
      continue
    for replica_idx in range(taskgraph.local_num_replicas):
      bcast_variables = taskgraph.get_variables(replica_idx)
      if zero_enabled():
        # If zero is enabled, each replica maintains its own states.
        bcast_variables = [v for v in bcast_variables if not is_apply_related(v)]
        tf_logging \
            .info("Broadcast weight only because zero is enabled.")
      bcast_variables = [v for v in bcast_variables if v.dtype not in [dtypes.resource, dtypes.resource_ref]]
      if not bcast_variables:
        continue
      num_splits = estimate_split_num_for_comm(bcast_variables)
      with ops.device(taskgraph.virtual_device.local_devices[replica_idx]):
        comm = create_serial_communicator(
            name="BROADCAST_{}".format(taskgraph_idx),
            devices=taskgraph.virtual_device.all_devices,
            max_splits=num_splits)
        reduced_variables = comm.broadcast(bcast_variables)
        for idx, variable in enumerate(bcast_variables):
          assign_ops.append(state_ops.assign(variable, reduced_variables[idx]))
  return control_flow_ops.group(assign_ops)


def _append_replicated_fetches(fetch, replicas):
  """Append replicated node to fetches. The fetches argument may be a single
  graph element, or an arbitrarily nested list, tuple, namedtuple, dict, or
  OrderedDict containing graph elements at its leaves. A graph element can
  be one of the following types: tf.Operation, tf.Tensor, tf.SparseTensor,
  get_tensor_handle, string.See tf.Session.run for more details."""
  if isinstance(fetch, list):
    for index, item in enumerate(fetch):
      fetch[index] = _append_replicated_fetches(item, replicas)
    return fetch
  elif isinstance(fetch, (dict, OrderedDict)):
    for key in fetch:
      fetch[key] = _append_replicated_fetches(fetch[key], replicas)
    return fetch
  elif isinstance(fetch, tuple):
    for index, item in enumerate(fetch):
      res = _append_replicated_fetches(item, replicas)
      if id(item) != id(res):
        fetch = update_tuple(fetch, res, index)
    return fetch
  elif isinstance(fetch, (ops.Tensor, RefVariable)):
    graph = Graph.get()
    taskgraph = graph.get_tensor_by_name(fetch.name).taskgraph
    if taskgraph is None:
      return fetch
    if taskgraph.strategy_context.split_strategy:
      return fetch
    tensor_replicas = graph.get_local_replicas(graph.get_tensor_by_name(fetch.name))
    merged_name = fetch.name + constant.MERGED_REPLICAS_SUFFIX
    if merged_name in graph.merged_outputs_map:
      replicas.extend(graph.merged_outputs_map[merged_name])
    else:
      replicas.extend([item.primitive_obj for item in tensor_replicas])
    if fetch.name in graph.merged_outputs_map:
      return graph.merged_outputs_map[fetch.name]
    return fetch
  elif isinstance(fetch, ops.Operation):
    graph = Graph.get()
    op_replicas = graph.get_local_replicas(graph.get_operation_by_name(fetch.name))
    replicas.extend([item.primitive_obj for item in op_replicas])
    return fetch
  elif isinstance(fetch, (six.string_types, SparseTensor)):
    # TODO(wangang.wa): String and SparseTensor should be supported.
    return fetch
  else:
    raise ValueError("Type of fetches is not supported for epl now. "
                     "Fetch type: %s." % type(fetch))


def _init_local_resources(self, fn):
  """Try to init local resources if needed."""
  Graph.get().set_model_phase(ModelPhase.SESSION_RUN_PHASE)

  assign_ops = None
  if not Graph.get().is_local_resources_ready and Env.get().cluster:
    if Graph.get().need_parallel:
      assign_ops = broadcast_variables()
    local_resources = resources.local_resources()
    Env.get().parallel_information[constant.ALL_COMM_RESOURCES].extend(local_resources)
    # Initialize local_resources for gradients aggregation and
    # variables broadcasting.
    Graph.get().is_local_resources_ready = True
    local_resources_init_op = control_flow_ops.group(resources.initialize_resources(local_resources))
    fn(self, local_resources_init_op)
    # Initialize local variables.
    local_variables_init_op = tf_variables.local_variables_initializer()
    fn(self, local_variables_init_op)
  return assign_ops


def replace_logging_tensor_hook(logging_hooks):
  """Replace tensor in logging_hooks with merged tensor if any."""
  for log_hook in logging_hooks:
    matched_tensor = {}
    for log_key, tensor in log_hook[1].items():
      merged_tensor = Graph.get().merged_outputs_map.get(tensor.name)
      if merged_tensor is not None:
        matched_tensor[log_key] = merged_tensor
    for key, tensor in matched_tensor.items():
      log_hook[1][key] = tensor


def base_session_run(fn):
  """Initialize local resource and broadcast variables after variables init."""
  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    """Base session run."""
    # Do not need to check replicas, initialize local resources and merge
    # outputs for some scenes which like running init ops in saver restore
    # function. Just run fetches and return results is ok. Only fetches
    # runned by users need to go through epl runing hooks.
    if Env.get().parallel_information and Env.get().parallel_information.get(constant.INFO_KEY_PREVENT_RUN_HOOK):
      return fn(self, fetches, feed_dict, options, run_metadata)
    ops.get_default_graph()._finalized = False
    assign_ops = _init_local_resources(self, fn)

    if isinstance(fetches, dict) and "caller" not in fetches:
      tf_logging.warn("This is not a MonitoredSession for no key caller in dict fetches: ", fetches)
      return fn(self, fetches, feed_dict, options, run_metadata)
    actual_fetches = fetches if isinstance(fetches, dict) else {"caller": fetches}
    actual_fetches["replicas"] = []
    if Graph.get().need_parallel:
      logging_hooks = [(k, hooks) for k, hooks in actual_fetches.items() if k.__class__.__name__ == 'LoggingTensorHook']
      replace_logging_tensor_hook(logging_hooks)
      actual_fetches["caller"] = _append_replicated_fetches(actual_fetches["caller"], actual_fetches["replicas"])
    outputs = fn(self, actual_fetches, feed_dict, options, run_metadata)

    if assign_ops:
      fn(self, assign_ops)
    ops.get_default_graph()._finalized = True

    return outputs if isinstance(fetches, dict) else outputs["caller"]

  return run


def base_session_makecallable(fn):
  """Hook session.make_callable."""
  def make_callable(self, fetches, feed_list=None, accept_options=False):
    """Basic session make_callable."""
    if isinstance(fetches, dict) and "caller" not in fetches:
      raise ValueError("This is not a MonitoredSession/Monitored"
                       "TrainingSession because of no key 'caller' "
                       "in dict fetches.")
    actual_fetches = fetches if isinstance(fetches, dict) else {"caller": fetches}
    actual_fetches["replicas"] = []
    actual_fetches["caller"] = _append_replicated_fetches(actual_fetches["caller"], actual_fetches["replicas"])
    outputs = fn(self, actual_fetches, feed_list, accept_options)
    return outputs

  return make_callable


def function_add_to_graph(fn):
  """Hook add_to_graph function of tensorflow function."""
  def add_to_graph(self, *args, **kwargs):
    pre_function_name = Graph.get().current_function_name
    Graph.get().current_function_name = self._func_name
    with ModelPhase(ModelPhase.ADD_FUNCTION):
      fn(self, *args, **kwargs)
    Graph.get().current_function_name = pre_function_name

  return add_to_graph


def _is_func_dataset_related():
  """Check if the function is created by dataset."""
  call_stacks = inspect.stack()
  for call_stack in call_stacks:
    if any(fn in call_stack.filename for fn in constant.dataset_related_files):
      return True
  return False


def func_graph_create_op(fn):
  """Hook create_op function of FuncGraph to get tensorflow function."""
  def create_op(self, *args, **kwargs):
    """Create FuncGraph op. Put dataset-related ops to CPU."""
    pre_function_name = Graph.get().current_function_name
    if _is_func_dataset_related():
      cpu_device = Env.get().cluster.current_worker_cpu()
      with ModelPhase(ModelPhase.ADD_FUNCTION), ops.device(cpu_device):
        res = fn(self, *args, **kwargs)
    else:
      with ModelPhase(ModelPhase.ADD_FUNCTION):
        res = fn(self, *args, **kwargs)
    Graph.get().current_function_name = pre_function_name
    return res

  return create_op


def saver_init(fn):
  """Hook saver init to select out save_and_restore_operations."""
  def init(self, *args, **kwargs):
    """Tensorflow saver init function."""
    with ModelPhase(ModelPhase.SAVE_AND_RESTORE):
      ret = fn(self, *args, **kwargs)
    return ret

  return init


def saver_save(fn):
  """Only first constucting worker allowed to save checkpoint."""
  def save(self, *args, **kwargs):
    """Tensorflow saver save function."""
    # TODO(wangang.wa): This code will be removed after merging
    # variables for split strategy.
    for taskgraph in Graph.get().taskgraphs:
      if taskgraph.strategy_context.split_strategy:
        with ModelPhase(ModelPhase.SAVE_AND_RESTORE):
          ret = fn(self, *args, **kwargs)
          return ret

    if Graph.get().first_constructor_rank != Env.get().cluster.worker_index:
      return None

    with ModelPhase(ModelPhase.SAVE_AND_RESTORE):
      ret = fn(self, *args, **kwargs)
      return ret

  return save


def saver_restore(fn):
  """Hook tensorflow saver restore function. Only first constructing worker
  allowed to restore checkpiont."""
  def restore(self, sess, save_path):
    """Tensorflow saver restore function."""
    # Initialize replicated variables on current worker.
    Env.get().parallel_information[constant.INFO_KEY_PREVENT_RUN_HOOK] = True
    graph = Graph.get()
    sess.run(graph.primitive_init_op)
    if Graph.get().need_parallel:
      init_replicas = graph.get_local_replicas(
          graph.get_operation_by_name(graph.primitive_init_op.name))
      for init in init_replicas:
        sess.run(init.primitive_obj)
    Env.get().parallel_information[constant.INFO_KEY_PREVENT_RUN_HOOK] = False

    ret = None
    # TODO(wangang.wa): This code will be removed after merging
    # variables for split strategy.
    if Graph.get().first_constructor_rank == Env.get().cluster.worker_index or \
        any(taskgraph.strategy_context.split_strategy is not None for taskgraph in Graph.get().taskgraphs):
      with ModelPhase(ModelPhase.SAVE_AND_RESTORE):
        ret = fn(self, sess, save_path)
    return ret

  return restore


def summary_scalar(fn):
  """Hook tf.summary.scalar"""
  def scalar(name, tensor, collections=None, family=None):
    val = fn(name, tensor, collections, family)
    Graph.get().summary_map[val.name] = SummaryInfo(name, tensor.name, constant.SUMMARY_SCALAR_TYPE)
    return val

  return scalar


def summary_image(fn):
  """Hook tf.summary.image"""
  def image(name, tensor, max_outputs=3, collections=None, family=None):
    val = fn(name, tensor, max_outputs, collections, family)
    Graph.get().summary_map[val.name] = SummaryInfo(name, tensor.name, constant.SUMMARY_IMAGE_TYPE)
    return val

  return image


def summary_histogram(fn):
  """Hook tf.summary.histogram"""
  def histogram(name, values, collections=None, family=None):
    val = fn(name, values, collections, family)
    Graph.get().summary_map[val.name] = SummaryInfo(name, values.name, constant.SUMMARY_HISTOGRAM_TYPE)
    return val

  return histogram


def summary_audio(fn):
  """Hook tf.summary.audio"""
  def audio(name,
            tensor,
            sample_rate,
            max_outputs=3,
            collections=None,
            family=None):
    """audio func."""
    val = fn(name, tensor, sample_rate, max_outputs, collections, family)
    Graph.get().summary_map[val.name] = SummaryInfo(name, tensor.name, constant.SUMMARY_AUDIO_TYPE)
    return val

  return audio


def summary_text(fn):
  """Hook tf.summary.text"""
  def text(name, tensor, collections=None):
    val = fn(name, tensor, collections)
    Graph.get().summary_map[val.name] = SummaryInfo(name, tensor.name, constant.SUMMARY_TEXT_TYPE)
    return val

  return text


def summary_tensor(fn):
  """Hook tf.summary.tensor_summary"""
  def tensor_summary(name,
                     tensor,
                     summary_description=None,
                     collections=None,
                     summary_metadata=None,
                     family=None,
                     display_name=None):
    """Record relation between tensor and summary tag."""
    val = fn(name, tensor, summary_description, collections, summary_metadata,
             family, display_name)
    Graph.get().summary_map[val.name] = SummaryInfo(name, tensor.name, constant.SUMMARY_TENSOR_TYPE)
    return val

  return tensor_summary


def distributed_add_weight(fn):
  """Replace add_weight with distributed_add_weight for Split context."""
  def add_weight(self,
                 name=None,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 constraint=None,
                 partitioner=None,
                 use_resource=None,
                 synchronization=tf_variables.VariableSynchronization.AUTO,
                 aggregation=tf_variables.VariableAggregation.NONE,
                 **kwargs):
    """Re-implementation of add_weight."""
    strategy = Env.get().strategy_context.split_strategy
    if strategy and shape:
      devices = strategy.devices
      fan_in, fan_out = init_ops._compute_fans(shape)
      initializer = initializers.get(initializer, fan_in=fan_in, fan_out=fan_out)
      shape = list(shape)
      num_devices = len(devices)
      shape[0] = dispatch_across_consumers(shape[0], num_devices, Env.get().cluster.worker_index)
      shape = tuple(shape)
    res = fn(self,
             name=name,
             shape=shape,
             dtype=dtype,
             initializer=initializer,
             regularizer=regularizer,
             trainable=trainable,
             constraint=constraint,
             partitioner=partitioner,
             use_resource=use_resource,
             synchronization=synchronization,
             aggregation=aggregation,
             **kwargs)
    return res

  return add_weight


def distributed_dense_layer(fn):
  """Replace dense layer to distributed dense layer for Split context."""
  def dense(*args, **kwargs):
    """Re-implementation of dense."""
    strategy = Env.get().strategy_context.split_strategy
    if strategy and not strategy.is_nested:
      strategy.is_nested = True
      res = distributed_dense(*args, **kwargs)
      strategy.is_nested = False
    else:
      res = fn(*args, **kwargs)
    return res

  return dense


def distributed_sparse_softmax_cross_entropy(fn):
  """Replace sparse softmax cross entropy loss function for Split context."""
  def sparse_softmax_cross_entropy(*args, **kwargs):
    """Re-implementation of sparse_softmax_cross_entropy."""
    strategy = Env.get().strategy_context.split_strategy
    if strategy and not strategy.is_nested:
      strategy.is_nested = True
      res = distributed_sparse_softmax_cross_entropy_with_logits(*args, **kwargs)
      strategy.is_nested = False
    else:
      res = fn(*args, **kwargs)
    return res

  return sparse_softmax_cross_entropy


def distributed_argmax(fn):
  """Replace argmax function for Split context."""
  def argmax(*args, **kwargs):
    """Re-implementation of argmax."""
    strategy = Env.get().strategy_context.split_strategy
    if strategy and not strategy.is_nested:
      strategy.is_nested = True
      res = distributed_ops.distributed_argmax(*args, **kwargs)
      strategy.is_nested = False
    else:
      res = fn(*args, **kwargs)
    return res

  return argmax


def distributed_einsum(fn):
  """Replace einsum with distributed_einsum for Split context."""
  def einsum(equation, *inputs, **kwargs):
    """Re-implemention of einsum."""
    strategy = Env.get().strategy_context.split_strategy
    if strategy:
      # TODO(jiangle.jl): Remove when epl supports split nested with replica
      assert len(Env.get().cluster.virtual_devices) == 1
      inputs = list(inputs)
      devices = strategy.devices
      num_devices = len(devices)
      if num_devices > 1:
        current_device = common.get_device_string(task=Env.get().cluster.worker_index)

        if constant.INFO_EINSUM_INDEX not in Env.get().parallel_information:
          if constant.SHARED_COMMUNICATOR_FOR_DISPATCH_AND_COMBINE in Graph.get().collective_communicator:
            comm = Graph.get().collective_communicator[constant.SHARED_COMMUNICATOR_FOR_DISPATCH_AND_COMBINE]
          else:
            comm = create_simple_communicator(constant.SHARED_COMMUNICATOR_FOR_DISPATCH_AND_COMBINE, devices)
            Graph.get().collective_communicator[constant.SHARED_COMMUNICATOR_FOR_DISPATCH_AND_COMBINE] = comm
          # TODO(jiangle.jl): Refactor when epl supports auto split
          Env.get().parallel_information[constant.INFO_EINSUM_INDEX] = 1
          inputs[0] = alltoall(comm, inputs[0], current_device)
          inputs[0] = array_ops.concat(array_ops.split(inputs[0], num_devices, 0), axis=2)
        else:
          # TODO(jiangle.jl): Refactor when epl supports auto split
          Env.get().parallel_information[constant.INFO_EINSUM_INDEX] += 1
          if not Env.get().parallel_information[constant.INFO_EINSUM_INDEX] % constant.NUM_EINSUM_IN_SPLIT_FOR_MOE:
            assert constant.SHARED_COMMUNICATOR_FOR_DISPATCH_AND_COMBINE in Graph.get().collective_communicator, \
                "Combine tensor should use the same communicator as dispatching"
            comm = Graph.get().collective_communicator[constant.SHARED_COMMUNICATOR_FOR_DISPATCH_AND_COMBINE]
            del Env.get().parallel_information[constant.INFO_EINSUM_INDEX]
            inputs[1] = array_ops.concat(array_ops.split(inputs[1], num_devices, 2), axis=0)
            inputs[1] = alltoall(comm, inputs[1], current_device)
    return fn(equation, *inputs, **kwargs)

  return einsum


def distributed_equal(fn):
  """Replace argmax function for Split context."""
  def equal(x, y, name=None):
    """Re-implementation of equal."""
    strategy = Env.get().strategy_context.split_strategy
    if strategy and not strategy.is_nested:
      strategy.is_nested = True
      res = distributed_ops.distributed_equal(x, y, name=name)
      strategy.is_nested = False
    else:
      res = fn(x, y, name=name)
    return res

  return equal


def add_layer_hooks():
  """Add hooks of layers to replace function for Split context
  in in-place mode."""
  # TODO(wangang.wa): Hook __init__ function and apply function of Dense
  # class is a better way which could support user denfine Dense object
  # first and then execute its call function.
  layers.dense = distributed_dense_layer(layers.dense)
  losses.sparse_softmax_cross_entropy = distributed_sparse_softmax_cross_entropy(losses.sparse_softmax_cross_entropy)
  gen_math_ops.arg_max = distributed_argmax(gen_math_ops.arg_max)
  gen_math_ops.equal = distributed_equal(gen_math_ops.equal)
  math.equal = distributed_equal(math.equal)
  math_ops.equal = distributed_equal(math_ops.equal)
  tensorflow.equal = distributed_equal(tensorflow.equal)
  base.base_layer.Layer.add_weight = distributed_add_weight(base.base_layer.Layer.add_weight)
  tensorflow.einsum = distributed_einsum(tensorflow.einsum)
  linalg.einsum = distributed_einsum(linalg.einsum)


def get_group_name(name, iid=None):
  """Get distinct func name for op group.

    Args:
      name: group name.
      iid: identifier of group."""
  # Keep distinct function/class for op group.
  group_counter = Env.get().parallel_information.get("GROUP_COUNTER")
  if group_counter is None:
    group_counter = Counter()
    Env.get().parallel_information["GROUP_COUNTER"] = group_counter
  if iid is None:
    group_counter[name] += 1
    iid = group_counter[name]
  return "{}_{}".format(name, iid)


def set_op_group(group_name):
  Env.get().parallel_information["CURRENT_OP_GROUP"] = group_name


def get_op_group():
  if "CURRENT_OP_GROUP" not in Env.get().parallel_information:
    set_op_group(None)
  return Env.get().parallel_information["CURRENT_OP_GROUP"]


def mark_ops(fn):
  """Mark ops."""
  def internel(*args, **kwargs):
    """Hook ops to add group info."""
    # In case of nested calls, use the outer group.
    if hasattr(fn, "__name__"):
      group_name = get_group_name(fn.__name__)
    else:
      group_name = get_group_name(str(fn))
    if get_op_group() is None:
      set_op_group(group_name)
    ret = fn(*args, **kwargs)
    if get_op_group() == group_name:
      set_op_group(None)
    return ret

  return internel


def mark_keras_layer(fn):
  """Mark the begin and end of base layer. In order to group related ops."""
  def hook_keras_layer(self, *args, **kwargs):
    """Add keras layer group hook."""
    # Ignore user defined layers.
    if not self.__class__.__name__ in tensorflow.keras.layers.__dict__.keys():
      return fn(self, *args, **kwargs)
    group_name = get_group_name(self.name, id(self))
    if get_op_group() is None:
      set_op_group(group_name)

    ret = fn(self, *args, **kwargs)
    if get_op_group() == group_name:
      set_op_group(None)
    return ret

  return hook_keras_layer


def mark_optimizer(fn):
  def hook_optimizer_init(self, *args, **kwargs):
    apply_method_code = self.__class__.apply_gradients.__code__
    base_method_code = self.__class__.__base__.apply_gradients.__code__
    if apply_method_code != base_method_code:
      self.__class__.apply_gradients = optimizer_apply_gradients(self.__class__.apply_gradients)
    return fn(self, *args, **kwargs)
  return hook_optimizer_init


def estimator_call_model_fn(call_fn):
  def hook_call_model_fn(self, *args, **kwargs):
    mode = args[2]
    Graph.get().set_model_mode(mode)
    return call_fn(self, *args, **kwargs)

  return hook_call_model_fn


def _sync_signal():
  """Sync a tensor among constructor workers."""
  all_devices = Env.get().cluster.virtual_devices[0].all_devices
  all_devices = [device for device in all_devices if 'GPU:0' in device]
  if len(all_devices) <= 1:
    return
  with ops.Graph().as_default():
    signal = constant_op.constant([1])
    with ops.device(Env.get().cluster.current_worker_chief_gpu()):
      comm = create_serial_communicator(name="BROADCAST_SIGNAL", devices=all_devices)
      sync_tensor = comm.broadcast([signal])
    with monitored_session.MonitoredTrainingSession() as sess:
      sess.run(sync_tensor)


def estimator_training_evaluate_hook():
  """Overwrite estimator evaluate."""
  def hook_estimator_evaluate(self, global_step_value):
    """Hook estimator evaluate fn."""

    self._timer.update_last_triggered_step(global_step_value)
    self.eval_result, self.export_results = (self._evaluator.evaluate_and_export())
    # Avoid session creation fail for last evaluation.
    _sync_signal()
    if self.eval_result.status != training._EvalStatus.EVALUATED:
      # Use warning instead of error as only worker0 evaluates.
      tf_logging.warn('There was no new checkpoint after the training. '
                      'Eval status: {}'.format(self.eval_result.status))

  return hook_estimator_evaluate


def get_device_fn_for_estimator():
  def device_function(op):
    worker_device = '/job:worker/task:%d' % (Env.get().cluster.worker_index)
    current_device = pydev.DeviceSpec.from_string(op.device or "")
    worker_device = pydev.DeviceSpec.from_string(worker_device)
    worker_device.merge_from(current_device)
    return worker_device.to_string()
  return device_function


def estimator_run_config(fn):
  """Hook estimator RunConfig"""

  def run_config_init(self, *args, **kwargs):
    """Ignore distributed strategy set by users."""
    tf_config = os.environ.get(constant.ENV_TF_CONFIG)
    # Clear TF_CONFIG, so that it will not trigger distributed setup in Estimator.
    if tf_config:
      os.environ[constant.ENV_TF_CONFIG] = "{}"
    if "train_distribute" in kwargs:
      dist_strategy = kwargs.pop("train_distribute")
      tf_logging.warn("Ignore distributed training strategy {} set in estimator when EPL is enabled.".format(dist_strategy))
    if "device_fn" in kwargs:
      kwargs.pop("device_fn")
      tf_logging.warn("Ignore device_fn set in estimator when EPL is enabled.")
    kwargs["device_fn"] = get_device_fn_for_estimator()
    res = fn(self, *args, **kwargs)
    # Set TF_CONFIG back.
    if tf_config:
      os.environ[constant.ENV_TF_CONFIG] = tf_config
    return res
  return run_config_init


def add_op_group_hooks():
  """Add op group hooks"""
  ops_list = [tensorflow.nn, tensorflow.losses, tensorflow.linalg]
  for operations in ops_list:
    for name, fn in operations.__dict__.items():
      if name.startswith('_'): continue
      if isinstance(fn, types.FunctionType):
        operations.__dict__[name] = mark_ops(fn)
  tensorflow.while_loop = mark_ops(tensorflow.while_loop)
  control_flow_ops.while_loop = mark_ops(control_flow_ops.while_loop)
  Layer.__call__ = mark_keras_layer(Layer.__call__)


def add_hooks():
  """Add epl hooks."""
  # Make sure add_hooks is only called once.
  global IS_EPL_HOOKED
  if IS_EPL_HOOKED:
    tf_logging.warn(
        "EPL add_hooks should be called only once, will ignore this call.")
    return
  add_op_group_hooks()
  IS_EPL_HOOKED = True
  if Version(__version__) >= Version("1.12.0") and Version(__version__) < Version("1.14.0"):
    # for tensorflow 1.12/1.13, _MaybeCompile and _GradientsHelper defined in gradients_impl.
    from tensorflow.python.ops import gradients_impl
    control_flow_ops.ControlFlowState.AddWhileContext = control_flow_add_while_context(control_flow_ops.ControlFlowState.AddWhileContext)
    gradients_impl._GradientsHelper = gradients_impl_gradients_helper(gradients_impl._GradientsHelper)
    gradients_impl._MaybeCompile = gradients_impl_maybe_compile(gradients_impl._MaybeCompile)

  elif Version(__version__) < Version("2.0"):
    # for tensorflow 1.14/1.15, _MaybeCompile and _GradientsHelper defined in gradients_util.
    from tensorflow.python.ops import control_flow_state
    from tensorflow.python.ops import gradients_util
    from tensorflow.python.framework import func_graph
    control_flow_state._ControlFlowState.AddWhileContext = control_flow_add_while_context(control_flow_state._ControlFlowState.AddWhileContext)
    gradients_util._GradientsHelper = gradients_impl_gradients_helper(gradients_util._GradientsHelper)
    gradients_util._MaybeCompile = gradients_impl_maybe_compile(gradients_util._MaybeCompile)
    func_graph.FuncGraph.create_op = func_graph_create_op(func_graph.FuncGraph.create_op)
  else:
    raise RuntimeError("Version of tensorflow is not supported for now. Tenosrflow Version: %s." % __version__)

  ops.Graph._add_op = graph_add_operation(ops.Graph._add_op)
  ops.Graph.__init__ = graph_init(ops.Graph.__init__)
  ops.Graph.finalize = graph_finalize(ops.Graph.finalize)
  optimizer.Optimizer.apply_gradients = optimizer_apply_gradients(optimizer.Optimizer.apply_gradients)
  session.BaseSession.__init__ = base_session_init(session.BaseSession.__init__)
  session.BaseSession.make_callable = base_session_makecallable(session.BaseSession.make_callable)
  session.BaseSession.run = base_session_run(session.BaseSession.run)
  monitored_session.Scaffold.finalize = scaffold_finalize(monitored_session.Scaffold.finalize)
  function._DefinedFunction.add_to_graph = function_add_to_graph(function._DefinedFunction.add_to_graph)
  saver.Saver.__init__ = saver_init(saver.Saver.__init__)
  saver.Saver.save = saver_save(saver.Saver.save)
  saver.Saver.restore = saver_restore(saver.Saver.restore)
  monitored_session._MonitoredSession.__init__ = monitored_session_init(monitored_session._MonitoredSession.__init__)
  variable_scope.VariableScope.__init__ = variable_scope_init(variable_scope.VariableScope.__init__)
  summary.scalar = summary_scalar(summary.scalar)
  summary.image = summary_image(summary.image)
  summary.histogram = summary_histogram(summary.histogram)
  summary.audio = summary_audio(summary.audio)
  summary.text = summary_text(summary.text)
  summary.tensor_summary = summary_tensor(summary.tensor_summary)

  optimizer.Optimizer.__init__ = mark_optimizer(optimizer.Optimizer.__init__)

  add_layer_hooks()
  estimator.Estimator._call_model_fn = estimator_call_model_fn(estimator.Estimator._call_model_fn)
  run_config.RunConfig.__init__ = estimator_run_config(run_config.RunConfig.__init__)
  training._NewCheckpointListenerForEvaluate._evaluate = estimator_training_evaluate_hook()

# pylint: enable=protected-access
