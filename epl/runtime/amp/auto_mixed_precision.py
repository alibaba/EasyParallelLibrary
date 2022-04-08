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
"""Memory Efficient automatic mixed precision."""

from collections import deque
from tensorflow.python.platform import tf_logging
from tensorflow.core.framework import types_pb2, attr_value_pb2
from tensorflow.python.framework import ops as tfops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.versions import __version__

from epl.env import Env
from epl.ir.graph import Graph
from epl.ir.graph import ModelPhase
from epl.parallel.ops import node_clone_for_amp
from epl.parallel.ops import Colocate
from epl.utils import constant
from epl.utils.common import in_while_loop


# Numerically-safe (for execution in fp16), always converted as fp16.
allow_list = ["BlockLSTM", "BlockLSTMV2", "BlockLSTMGrad", "BlockLSTMGradV2",
              "Conv2D", "Conv2DBackpropFilter", "Conv2DBackpropInput",
              "CudnnRNN", "CudnnRNNBackprop", "CudnnRNNBackpropV2",
              "CudnnRNNBackpropV3", "CudnnRNNV2", "CudnnRNNV3", "Einsum",
              "FusedConv2DBiasActivation", "GRUBlockCell", "GRUBlockCellGrad",
              "LSTMBlockCell", "LSTMBlockCellGrad", "MatMul", "BatchMatMul",
              "BatchMatMulV2"]

# Numerically-dangerous, keep fp32 computation.
deny_list = ["Exp", "Expm1", "L2Loss", "Log", "Log1p", "LogSoftmax", "Mean",
             "Pow", "SaveV2", "Softmax", "SoftmaxCrossEntropyWithLogits",
             "SparseSoftmaxCrossEntropyWithLogits", "Sum"]

# Numerically-safe, but which may be made unsafe by an upstream blacklist op.
gray_list = ["Add", "AddN", "AddV2", "AvgPool", "AvgPool3D", "AvgPool3DGrad",
             "AvgPoolGrad", "BiasAdd", "BiasAddGrad", "BiasAddV1", "Elu",
             "EluGrad", "Erf", "Erfc", "FloorDiv", "FusedBatchNormV2",
             "FusedBatchNormGradV2", "FusedBatchNormV3", "FusedBatchNormGradV3",
             "_FusedBatchNormEx", "Inv", "LeakyRelu", "LeakyReluGrad", "Mul",
             "Prod", "RealDiv", "Reciprocal", "Sigmoid", "SigmoidGrad",
             "Softplus", "SoftplusGrad", "Sqrt", "Sub", "Tanh", "TanhGrad"]

# Do not have numerically-significant effects, convert when necessary.
clear_list = ["Abs", "ArgMax", "ArgMin", "BatchToSpace", "BatchToSpaceND",
              "BroadcastTo", "Ceil", "CheckNumerics", "ClipByValue", "Concat",
              "ConcatV2", "DepthToSpace", "DynamicPartition", "DynamicStitch",
              "Enter", "EnsureShape", "Equal", "Exit", "ExpandDims", "Fill",
              "Floor", "Gather", "GatherNd", "GatherV2", "Greater",
              "GreaterEqual", "Identity", "IdentityN", "IsFinite", "IsInf",
              "IsNan", "Less", "LessEqual", "Max", "MaxPool", "MaxPool3D",
              "MaxPool3DGrad", "MaxPool3DGradGrad", "MaxPoolGrad",
              "MaxPoolGradGrad", "MaxPoolGradGradV2", "MaxPoolGradV2",
              "MaxPoolV2", "Maximum", "Merge", "Min", "Minimum", "MirrorPad",
              "MirrorPadGrad", "Neg", "NextIteration", "NotEqual", "OneHot",
              "OnesLike", "Pack", "Pad", "PadV2", "PreventGradient", "Rank",
              "Relu", "Relu6", "Relu6Grad", "ReluGrad", "Reshape",
              "ResizeNearestNeighbor", "ResizeNearestNeighborGrad", "Reverse",
              "ReverseSequence", "ReverseV2", "Round", "Select", "Shape",
              "ShapeN", "Sign", "Size", "Slice", "Snapshot", "SpaceToBatch",
              "SpaceToBatchND", "SpaceToDepth", "Split", "SplitV", "Squeeze",
              "StackPopV2", "StackPushV2", "StopGradient", "StridedSlice",
              "StridedSliceGrad", "Switch", "TensorArrayConcatV3",
              "TensorArrayGatherV3", "TensorArrayReadV3",
              "TensorArrayScatterV3", "TensorArraySplitV3",
              "TensorArrayWriteV3", "Tile", "TopK", "TopKV2", "Transpose",
              "Where", "ZerosLike"]

# List of op types excluded in AMP conversion.
preserve_list = ["NoOp", "VariableV2", "ConcatV2", "Snapshot", "Mean", "Pack",
                 "Const", "LogicalNot", "IteratorV2", "IteratorGetNext",
                 "Where", "GatherV2"]


tf32 = dtypes.float32
tf16 = dtypes.float16
pb32 = types_pb2.DT_FLOAT
pb16 = types_pb2.DT_HALF


def amp_enabled():
  """Check if amp is enabled."""
  config = Env.get().config
  if config.amp.level.lower() == "o1":
    return True
  return False


def debug_logging(msg):
  """Log if amp_log is enabled."""
  if Env.get().config.amp.debug_log:
    tf_logging.info(msg)


def get_phase(op):
  """Get op phase."""
  return Graph.get().operations[op.name].phase


def cast_tensor(ts, dtype):
  """Cast tensor to type dtype."""
  return math_ops.cast(ts, dtype=dtype, \
                       name=ts.op.name + "_{}".format(dtype.name))


def get_node_type_key(op):
  """Get node_def type key."""
  ntype = None
  if 'T' in op.node_def.attr:
    ntype = 'T'
  if 'Tidx' in op.node_def.attr:
    ntype = 'Tidx'
  return ntype


def is_float_op(op):
  """Check if operation is float operation."""
  type_key = get_node_type_key(op)
  if type_key:
    return op.node_def.attr[type_key].type == pb32
  return op.outputs[0].dtype == tf32


def is_read_variable_op(op):
  """Is variable read operation."""
  return op.type in ["Identity"] and \
      op.inputs[0].op.type in ["Variable", "VariableV2"]

def find_input_tensor_index(op, match_ts):
  matched = [i for i, in_ts in enumerate(op.inputs) if in_ts == match_ts]
  if matched:
    return matched[0]
  return -1

# pylint: disable=protected-access
class AMP(object):
  """
  Memory Efficient automatic mixed precision.
  """

  def __init__(self):
    self.whitelist = set()
    self.blacklist = set()
    self.cast_fp16_count = 0
    self.cast_fp32_count = 0

  def cast_inputs_fp16(self, op, control_cast=True):
    """Cast inputs of op to float 16."""
    for idx, inp in enumerate(op.inputs):
      if inp.dtype == tf32:
        inp_16 = cast_tensor(inp, tf16)
        debug_logging("Inserting cast to DT_HALF at {} {}" \
                      .format(inp.op.type, inp.name))
        if (is_read_variable_op(inp.op) or inp.op.type == "Const") \
            and control_cast:
          self.update_input_with_deps(op, idx, inp_16)
        else:
          self.cast_fp16_count += 1
          op._update_input(idx, inp_16)

  def convert(self):
    """Convert graph to mixed precision."""
    self.ops = self.filter_ops()
    tf_logging.info('Beginning pass 1')
    self.pass1_addwhite()
    tf_logging.info('Finished pass 1')
    tf_logging.info('Beginning pass 2')
    self.pass2_add_black()
    tf_logging.info('Finished pass 2')
    tf_logging.info('Beginning pass 3')
    self.pass3_clearnode()
    tf_logging.info('Finished pass 3')
    tf_logging.info('Beginning pass 4')
    self.pass4()
    tf_logging.info('Finished pass 4')
    self.mark_cast_outputwhite()
    self.cast_type()
    self.report_info()

  def should_process(self, op):
    return op in self.should_process_ops

  def filter_ops(self):
    ops = tfops.get_default_graph().get_operations()
    ops = [op for op in ops if get_phase(op) in \
           [ModelPhase.FORWARD, ModelPhase.BACKWARD]]
    filter_ops = [op for op in ops if op.type not in preserve_list]
    filter_ops = [op for op in ops if not in_while_loop(op)]
    self.should_process_ops = set(filter_ops)
    return ops

  def recognized(self, op):
    return op.type in allow_list or op.type in deny_list or \
           op.type in gray_list or op.type in clear_list

  def report_info(self):
    tf_logging.info("Total processable nodes: {}".format(len(self.ops)))
    recognized_ops = [op for op in self.ops if self.recognized(op)]
    tf_logging.info("Recognized nodes available for conversion: {}" \
                    .format(len(recognized_ops)))
    tf_logging.info("Whitelisted nodes converted: {}" \
                    .format(len(self.whitelist)))
    tf_logging.info("Blacklisted nodes blocking conversion: {}" \
                    .format(len(self.blacklist)))
    tf_logging.info("Total FP16 Cast ops used (excluding Const and Variable \
                    casts): {}".format(self.cast_fp16_count))

  def cast_node_fp16(self, op):
    """
    Cast operation node to float16.
    """
    if op.type == "Cast":
      if op.node_def.attr['SrcT'].type == pb32:
        op._set_attr('SrcT', attr_value_pb2.AttrValue(type=pb16))
        return op
    else:
      key = "T"
      if op.node_def.attr[key].type == pb32:
        epl_op = Graph.get().operations[op.name]
        with Colocate(epl_op):
          new_op = node_clone_for_amp(epl_op, tf16, pb16, constant.EPL_AMP_SUFFIX)
        # update output consumers
        for out_idx, out_ts in enumerate(op.outputs):
          for consumer in out_ts.consumers():
            idx = find_input_tensor_index(consumer, out_ts)
            consumer._update_input(idx, new_op.outputs[out_idx])
        return new_op
    return op

  def cast_type(self):
    """
    Cast operation/tensor types according to white/black list.
    """
    ops = tfops.get_default_graph().get_operations()
    for op in ops:
      if op in self.whitelist:
        self.cast_node_fp16(op)
        debug_logging("Changing type {} of {} node {} to DT_HALF" \
                      .format(get_node_type_key(op), op.type, op.name))
    ops = tfops.get_default_graph().get_operations()
    for op in ops:
      if op.node_def.attr['T'].type == pb16:
        if any(inp.dtype == tf32 for inp in op.inputs):
          self.cast_inputs_fp16(op)
    ops = tfops.get_default_graph().get_operations()
    for op in ops:
      for idx, inp in enumerate(op.inputs):
        if inp.dtype == tf16:
          if op.type == "Cast" and op.node_def.attr['SrcT'].type == pb32:
            op._set_attr('SrcT', attr_value_pb2.AttrValue(type=pb16))
          elif op.node_def.attr['T'].type == pb32:
            op._update_input(idx, cast_tensor(inp, tf32))
            debug_logging("Inserting cast to DT_FLOAT at {} {}" \
                          .format(inp.op.type, inp.name))

  def is_black(self, op):
    return op.type in deny_list

  def is_white(self, op):
    return op.type in allow_list

  def is_gray(self, op):
    return op.type in gray_list

  def is_clear(self, op):
    return op.type in clear_list


  def pass1_addwhite(self):
    """
    Add all performance-critical ops (aka "whitelist" ops) to the white_set.
    This is done under the assumption that whitelist ops are always
    numerically-safe in fp16 and that they are the most important ops for
    improving performance.
    :return:
    """
    for op in self.ops:
      if not self.should_process(op):
        continue
      if op.type in allow_list:
        if op not in self.whitelist:
          self.whitelist.add(op)
          self.paint_log(1, op, "WHITE")

  def op_consumers(self, op):
    """Get consumer ops."""
    res = []
    for out in op.outputs:
      res.extend(out.consumers())
    return res

  def add_inputs_if_meet(self, op, func, queue):
    """Add op inputs op to queue if meet func."""
    for inp in op.inputs:
      if func(inp.op):
        queue.append(inp.op)

  def add_outputs_if_meet(self, op, func, queue):
    """Add op outputs op to queue if meet func."""
    for con in self.op_consumers(op):
      if func(con):
        queue.append(con)

  def pass2_add_black(self):
    """
    Adds nodes to black_set iff they are on the blacklist or they are on a
    forward path from a blacklist node to a black/gray node (including the node
    at the end of the path) through clear and gray nodes.
    E.g., black -> gray -> clear -> gray -> clear -> white -> gray
    becomes: black -> black -> black -> black -> clear -> white -> gray.
    :return:
    """
    queue = deque()
    for o in self.ops:
      if self.is_black(o) or self.is_gray(o):
        queue.append(o)
    # Find clear nodes that are upstream of black or gray.
    upstream_of_black_or_gray_set = set()
    while queue:
      op = queue.pop()
      upstream_of_black_or_gray_set.add(op)
      for inp in op.inputs:
        iop = inp.op
        if iop not in upstream_of_black_or_gray_set and self.is_clear(iop) \
            and self.should_process(iop):
          queue.append(iop)
    # Propagate black forward through nodes in upstream_of_black_or_gray_set.
    queue = deque([o for o in self.ops if self.is_black(o) \
                   and o not in self.blacklist])
    while queue:
      op = queue.pop()
      if op not in self.blacklist:
        self.blacklist.add(op)
        self.paint_log(2, op, "BLACK")
      for con in self.op_consumers(op):
        if not self.op_consumers(con): continue
        if con in upstream_of_black_or_gray_set and \
            con not in self.blacklist and self.should_process(con):
          queue.append(con)

  def pass3_clearnode(self):
    """
    For all remaining nodes that are not considered dangerous (greylist
    and clearlist ops), find those that are between (i.e., both upstream
    and downstream of) white nodes, and add them to the white_set.
    This is done to avoid unnecessary casts between whitelist ops.
    :return:
    """
    downstream_ops = set()
    queue = deque([o for o in self.whitelist if self.should_process(o)])
    while queue:
      op = queue.pop()
      downstream_ops.add(op)
      cond = lambda con: self.should_process(con) and \
                         not self.is_white(con) and \
                         not con in self.blacklist and \
                         is_float_op(con) and \
                         (self.is_gray(con) or self.is_clear(con)) and \
                         con not in downstream_ops
      self.add_outputs_if_meet(op, cond, queue)
    queue = deque([o for o in self.whitelist if self.should_process(o)])
    upstream_ops = set()
    while queue:
      op = queue.pop()
      upstream_ops.add(op)
      if op not in self.whitelist:
        self.whitelist.add(op)
        self.paint_log(3, op, "WHITE")
      cond = lambda iop: iop in downstream_ops and iop not in upstream_ops
      self.add_inputs_if_meet(op, cond, queue)

  def paint_log(self, pass_num, op, color):
    """Print mark log."""
    ntype = get_node_type_key(op)
    debug_logging("Pass{}: Painting type {} of {} node {} {}" \
                  .format(pass_num, ntype, op.type, op.name, color))

  def pass4(self):
    """
    For all remaining clearlist nodes, add them to the white_set if they are
    connected to a node in the white_set via other clearlist nodes.
    This is done to increase the number of ops in the white_set without
    affecting numerical stability.
    :return:
    """
    clear_prop_set = set()
    queue = deque([op for op in self.whitelist if self.should_process(op)])
    while queue:
      op = queue.pop()
      clear_prop_set.add(op)
      if op not in self.whitelist:
        self.whitelist.add(op)
        self.paint_log(4, op, "WHITE")
      cond = lambda iop: not iop in self.whitelist\
                         and not iop in self.blacklist\
                         and self.should_process(iop)\
                         and is_float_op(iop) \
                         and self.is_clear(iop) \
                         and not is_read_variable_op(iop) \
                         and iop not in clear_prop_set
      self.add_inputs_if_meet(op, cond, queue)
      self.add_outputs_if_meet(op, cond, queue)

  def mark_cast_outputwhite(self):
    """
    This adds existing Cast nodes to white_set if all of their outputs are
    white, avoiding the need to add a new Cast node after an existing Cast.
    :return:
    """
    for op in self.ops:
      if op.type == "Cast":
        if all(self.is_white(con) for con in self.op_consumers(op)):
          self.whitelist.add(op)


  def update_input_with_deps(self, op, idx, new_input):
    op._update_input(idx, new_input)
    other_inp = [p.op for i, p in enumerate(op.inputs) if i != idx]
    if other_inp and not new_input.op.control_inputs:
      new_input.op._add_control_inputs(other_inp)
# pylint: enable=protected-access
