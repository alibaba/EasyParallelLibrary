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
"""save/restore tools."""

from collections import defaultdict
from distutils.version import LooseVersion as Version
import uuid

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework.versions import __version__
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.saver import BaseSaverBuilder

from epl.env import Env


if Version(__version__) < Version("1.15.0"):
  op_list_to_dict = BaseSaverBuilder.OpListToDict
  saveable_objects_for_op = BaseSaverBuilder.SaveableObjectsForOp
else:
  from tensorflow.python.training.saving import saveable_object_util # pylint: disable=ungrouped-imports
  op_list_to_dict = saveable_object_util.op_list_to_dict
  saveable_objects_for_op = saveable_object_util.saveable_objects_for_op


# pylint:disable=protected-access
class ShardingLoader(object):
  """loader to load sharded parameters."""

  def __init__(self, ckpt_file, var_list=None, assign_map=None,
               sharding_info=None):
    """Load a checkpoint with custom variable list, assignment map or sharding.
    Usage:
    ```
      loader = ShardingLoader(ckpt_file, var_list, assign_map, sharding_info)
      loader.restore()
    ```
    Args:
      ckpt_file: checkpoint file to load.
      var_list: a list of variables, if var_list is None,
        use all saveable variables as default.
      assign_map: mapping between variable name to checkpoint store name.
      sharding_info: mapping between variable name to sharding information.
        For a given variable, "begin" marks the starting indices in the
        checkpoint tensor. "size" represents the variable shape.
    """
    self.ckpt_file = ckpt_file
    if var_list:
      self.var_list = var_list
    else:
      # if var_list is not defined, get all variables.
      self.var_list = variables._all_saveable_objects()
    self.assign_map = assign_map if assign_map else {}
    self.sharding_info = sharding_info
    self.name2restore = {}
    self.name2init = defaultdict(list)

  def restore_op_ckpt(self, var, tensor_name):
    """Restore one op from checkpoint."""
    if tensor_name in self.name2restore:
      return self.name2restore[tensor_name]
    base_type = var.dtype.base_dtype
    spec = var._save_slice_info.spec if var._save_slice_info else ""
    restore_op = io_ops.restore_v2(self.ckpt_file, [tensor_name],
                                   [spec], [base_type],
                                   name="checkpoint_initializer")[0]
    if self.sharding_info and tensor_name in self.sharding_info:
      info = self.sharding_info[tensor_name]
      begin = info["begin"]
      size = info["size"]
      restore_op = array_ops.slice(restore_op, begin, size)

    restore_op.set_shape(var.shape)
    self.name2restore[tensor_name] = restore_op
    return restore_op

  def rewrite_init_op(self, variable, restore_op):
    """Replace init_op with checkpoint restore op."""
    names_to_saveables = op_list_to_dict([variable])
    saveable_objects = []
    for name, op in names_to_saveables.items():
      for s in saveable_objects_for_op(op, name):
        saveable_objects.append(s)
    assert len(saveable_objects) == 1  # Should be only one variable.
    init_op = saveable_objects[0].restore([restore_op], restored_shapes=None)

    variable._initializer_op = init_op
    return init_op

  def restore(self):
    """Add restore ops."""
    if self.ckpt_file is None:
      tf_logging.warn("ShardingLoader: Given checkpoint file is None")
      return
    tf_logging.info("ShardingLoader: restore checkpoint from {}" \
                    .format(self.ckpt_file))
    for var in self.var_list:
      # store_name is the name in ckpt file
      store_name = self.assign_map.get(var.op.name, var.op.name)
      restore_op = self.restore_op_ckpt(var, store_name)
      init_op = self.rewrite_init_op(var, restore_op)
      self.name2init[store_name].append(init_op)
    # Add control dependencies.
    last_deps = []
    for name, restore_op in self.name2restore.items():
      if last_deps:
        restore_op.op._add_control_inputs(last_deps)
      last_deps = [restore_op] + self.name2init[name]
      last_deps = [t.op for t in last_deps]

def _get_bytes(tensor):
  """Get tensor size in bytes.

  Args:
    tensor: Tensor: A input tensor.

  Returns:
    Size in bytes.
  """
  tensor_size = tensor.get_shape().num_elements() or 0
  return tensor_size * tensor.dtype.size

def _on_gpu(tensor):
  return 'gpu' in tensor.device.lower()

class MemoryEfficientBuilder(BaseSaverBuilder):
  """SaverBuilder with support for memory-efficient save."""

  def __init__(self, sharded_size=50*1024*1024):
    """Create a `MemoryEfficientBuilder`.

    Args:
      sharded_size: parameter size per shard
    """
    super(MemoryEfficientBuilder, self).__init__()
    self._save_bucket_size = sharded_size

  def _AddShardedSaveOpsForV2(self, checkpoint_prefix, per_device):
    sharded_suffix = "_temp_%s/part" % uuid.uuid4().hex
    tmp_checkpoint_prefix = string_ops.string_join(
        [checkpoint_prefix, sharded_suffix])

    num_shards = len(per_device)
    sharded_saves = []
    sharded_prefixes = []
    num_shards_tensor = constant_op.constant(num_shards, name="num_shards")
    last_save = control_flow_ops.no_op()
    cpu_device = Env.get().cluster.current_worker_cpu()
    for shard, (_, saveables) in enumerate(per_device):
      with ops.device(cpu_device):
        sharded_filename = self.sharded_filename(tmp_checkpoint_prefix, shard,
                                                 num_shards_tensor)
        sharded_prefixes.append(sharded_filename)
        with ops.control_dependencies([last_save]):
          shard_save = self._AddSaveOps(sharded_filename, saveables)
          last_save = shard_save
          sharded_saves.append(shard_save)

    with ops.control_dependencies([x.op for x in sharded_saves]):
      # Co-locates the merge step with the last device.
      with ops.device(cpu_device):
        # V2 format write path consists of a metadata merge step.  Once merged,
        # attempts to delete the temporary directory, "<user-fed prefix>_temp".
        merge_step = gen_io_ops.merge_v2_checkpoints(
            sharded_prefixes, checkpoint_prefix, delete_old_dirs=True)
        with ops.control_dependencies([merge_step]):
          # Returns the prefix "<user-fed prefix>" only.  DOES NOT include the
          # sharded spec suffix.
          return array_ops.identity(checkpoint_prefix)

  def _GroupByDevices(self, saveables):
    max_size = max(_get_bytes(s.op) for s in saveables if _on_gpu(s))
    bucket_size = max(max_size, self._save_bucket_size)
    tf_logging.info("save bucket size is {} bytes".format(bucket_size))
    grouped = [[]]
    acc = 0
    for saveable in saveables:
      if _on_gpu(saveable):
        tensor = saveable.op
        tensor_bytes = _get_bytes(tensor)
      else:
        tensor_bytes = 0
      if acc + tensor_bytes > bucket_size and grouped[-1]:
        grouped.append([])
        acc = 0
      grouped[-1].append(saveable)
      acc += tensor_bytes
    return [(g[0].device, g) for g in grouped]
# pylint:enable=protected-access
