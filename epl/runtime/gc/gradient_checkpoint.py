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
"""Gradient Checkpoint.
Code modified based on https://github.com/cybertronai/gradient-checkpointing"""

import contextlib
import tensorflow.contrib.graph_editor as ge
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging

from epl.env import Env

from epl.runtime.gc.auto_gradient_checkpoint import fast_backward_ops
from epl.runtime.gc.auto_gradient_checkpoint import is_consumer_in_while_loop
from epl.runtime.gc.auto_gradient_checkpoint import is_variable_const_related
from epl.runtime.gc.auto_gradient_checkpoint import search_checkpoint_tensors
from epl.runtime.gc.auto_gradient_checkpoint import toposort_ops
from epl.runtime.gc.auto_gradient_checkpoint import tf_toposort_tensors
from epl.runtime.gc.auto_gradient_checkpoint import tf_toposort
from epl.utils.constant import GC_AUTO
from epl.utils.constant import GC_AVOID_RECOMPUTE_OPS
from epl.utils.constant import GC_COLLECTION
from epl.utils.constant import GC_COLLECTION_NAME
from epl.utils.constant import GC_DST_SCOPE_NAME
from epl.utils.common import in_while_loop


_colocate_gradients_with_ops = True

RANDOM_OPS = ["RandomUniform", "Multinomial", "RandomStandardNormal",
              "ParameterizedTruncatedNormal", "TruncatedNormal"]


@contextlib.contextmanager
def capture_ops(scope_name):
  """Decorator to capture ops created in the block.
  with capture_ops() as ops:
    # create some ops
  print(ops) # => prints ops created.
  """
  op_list = []
  with ops.name_scope(scope_name):
    yield op_list

  g = ops.get_default_graph()
  op_list.extend(ge.select_ops(scope_name + "/.*", graph=g))


def is_random_op(op):
  """Check if op is random type."""
  return any(op.type == random_type for random_type in RANDOM_OPS)


# pylint: disable=protected-access
def tf_gradients(tf_gradient_function, *args, **kwargs):
  """Decorate tf.gradients calls with explicit device placement to avoid memory
  leaks when splitting model across multiple GPUs"""
  ys = args[0]
  source = ys[0] if isinstance(ys, (list, tuple)) else ys
  device = source.op.node_def.device if isinstance(source, ops.Tensor) else None
  with ops.device(device):
    return tf_gradient_function(*args, **kwargs)


def gradients(tf_gradient_function, checkpoint_type, *args, **kwargs):
  """Compute gradients with gradient checkpoint(activation recomputation).

  Args:
    tf_gradient_function: tensorflow gradient function.
    checkpoint_type: way to get a list of tensor to be re-use in backward pass.
                    'collection': look for a tensorflow collection named 'checkpoints'
                    'auto': search checkpoints tensor automatically.
  """
  ys = [args[0]] if not isinstance(args[0], list) else args[0]
  xs = [args[1]] if not isinstance(args[1], list) else args[1]
  grad_ys = args[2]

  gate_gradients = args[5]
  aggregation_method = args[6]
  stop_gradients = args[7]
  # Get operations from xs
  xs_ops = [x.op for x in xs]
  ys_ops = [y.op for y in ys]

  bwd_ops = ge.get_backward_walk_ops(ys_ops, inclusive=True)

  # Forward ops are all ops that are candidates for recomputation
  fwd_ops = ge.get_forward_walk_ops(xs_ops, inclusive=True, within_ops=bwd_ops)

  # Exclude ops with no inputs, and is variable-related.
  fwd_ops = [op for op in fwd_ops if op.inputs and (not op in xs_ops) and
             (not '/assign' in op.name.lower()) and (not '/read' in op.name)]
  # Get the forward tensors.
  ts_all = ge.filter_ts(fwd_ops, True)
  ts_all = [t for t in ts_all if '/read' not in t.name]
  ts_all = set(ts_all) - set(xs) - set(ys)

  # Get a list of tensors to reuse during forward pass
  tf_logging.info("Use checkpoint type: {}".format(checkpoint_type))
  if checkpoint_type == GC_COLLECTION:
    checkpoints = ops.get_collection(GC_COLLECTION_NAME)
  elif checkpoint_type == GC_AUTO:
    checkpoints = search_checkpoint_tensors(ts_all, ys, xs)
  else:
    raise Exception("{} is unsupported input for `checkpoint_type`".format(checkpoint_type))

  for t in ts_all:
    if t.op.type in GC_AVOID_RECOMPUTE_OPS:
      checkpoints.append(t)
  for t in checkpoints:
    if in_while_loop(t.op) or is_consumer_in_while_loop(t.op):
      checkpoints.remove(t)
      tf_logging.warn("Remove checkpoint {}, while loop is not supported as a GC tensor by now.".format(t))

  checkpoints = list(set(checkpoints).intersection(ts_all))

  # No need to include input xs as checkpoint, is is already processed.
  xs_in_checkpoints = set(xs).intersection(set(checkpoints))
  if xs_in_checkpoints:
    tf_logging.debug("Some input nodes are also checkpoint tensors: {}".format(xs_in_checkpoints))
  ys_in_checkpoints = set(ys).intersection(set(checkpoints))
  # saving an output node (ys) gives no benefit in memory while creating new edge cases, exclude them
  if ys_in_checkpoints:
    tf_logging.debug("Some output nodes are also checkpoint tensors: {}".format(ys_in_checkpoints))

  # remove initial and terminal nodes from checkpoints list if present
  checkpoints = list(set(checkpoints) - set(ys) - set(xs))

  with capture_ops('TEMP_GRADIENTS') as temp_bwd_ops:
    base_gradients = tf_gradients(tf_gradient_function, *args, **kwargs)
  bwd_inputs = [t for op in temp_bwd_ops for t in op.inputs]
  sorted_fwd_ts = tf_toposort_tensors(fwd_ops)

  # Filter tensors that do not contribute to backward graph from checkpoints.
  flattened_ts = [t for ts in sorted_fwd_ts for t in ts if not is_variable_const_related(t.op)]
  min_input_index = min(flattened_ts.index(t) for t in bwd_inputs if t in flattened_ts)

  ts_filtered = []
  for t in checkpoints:
    if flattened_ts.index(t) >= min_input_index:
      ts_filtered.append(t)
    else:
      tf_logging.warn("filter {} because it is not the input to bwd graph" \
                      .format(t))
  checkpoints = ts_filtered
  # check that we have some nodes to checkpoint
  if not checkpoints:
    tf_logging.warn('no checkpoints nodes found or given as input! \
                 Use default gradients.')
    return tf_gradients(tf_gradient_function, *args, **kwargs)
  Env.get().default_graph.gc_tensors = checkpoints
  for ind, t in enumerate(checkpoints):
    tf_logging.info("GC with tensor{}: {}".format(ind, t))

  # disconnect dependencies between checkpointed tensors
  checkpoints_disconnected = {}
  for x in checkpoints:
    if x.op and x.op.name is not None:
      grad_node = array_ops.stop_gradient(x, name=x.op.name + "_gc_sg")
    else:
      grad_node = array_ops.stop_gradient(x, name=x.op.name + "_gc_sg")
    grad_node.op._set_device(x.op.node_def.device)
    checkpoints_disconnected[x] = grad_node

  # partial derivatives to the checkpointed tensors and xs
  last_block_ops = fast_backward_ops(seed_ops=[y.op for y in ys],
                                     stop_at_ts=checkpoints,
                                     within_ops=fwd_ops)
  ge.reroute_ts(checkpoints_disconnected.values(),
                checkpoints_disconnected.keys(),
                can_modify=last_block_ops)

  # get gradients with respect to current boundary + original x's
  # copied_ys = [info._transformed_ops[y.op]._outputs[0] for y in ys]
  boundary = list(checkpoints_disconnected.values())
  dv = tf_gradients(tf_gradient_function, ys, boundary + xs, grad_ys,
                    "gradients", _colocate_gradients_with_ops, gate_gradients,
                    aggregation_method, stop_gradients, **kwargs)

  inputs_to_do_before = [y.op for y in ys]
  if grad_ys is not None:
    inputs_to_do_before += grad_ys

  # partial derivatives to the checkpointed nodes
  # dictionary of "node: backprop" for nodes in the boundary
  d_checkpoints = {
      r: dr
      for r, dr in zip(checkpoints_disconnected.keys(),
                       dv[:len(checkpoints_disconnected)])
  }

  # partial derivatives to xs (usually the params of the neural net)
  d_xs = dv[len(checkpoints_disconnected):]

  # incorporate derivatives flowing through the checkpointed nodes
  checkpoints_sorted_lists = tf_toposort(checkpoints, sorted_ts=sorted_fwd_ts)
  c_inp_xs = [[]] * len(xs)
  for ts in checkpoints_sorted_lists[::-1]:
    checkpoints_other = [r for r in checkpoints if r not in ts]
    checkpoints_disconnected_other = [
        checkpoints_disconnected[r] for r in checkpoints_other
    ]

    # copy part of the graph below current checkpoint node, stopping at
    # other checkpoints nodes
    ops_to_copy = fast_backward_ops(within_ops=fwd_ops,
                                    seed_ops=[r.op for r in ts],
                                    stop_at_ts=checkpoints_other)
    # Exclude dropout to handle randomness.
    ops_to_copy = [op for op in ops_to_copy if not is_random_op(op)]
    # Exclude copy of while loop
    ops_to_copy = [op for op in ops_to_copy if not in_while_loop(op)]
    if not ops_to_copy:  # we're done!
      tf_logging.info("GC stop early at {}".format(ts))
      break
    _, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {},
                                              dst_scope=GC_DST_SCOPE_NAME)
    for origin_op, op in info._transformed_ops.items():
      op._set_device(origin_op.node_def.device)
    copied_ops = info._transformed_ops.values()
    ge.reroute_ts(checkpoints_disconnected_other,
                  checkpoints_other,
                  can_modify=copied_ops)

    # gradient flowing through the checkpointed node
    boundary = []
    for r in ts:
      copied_op = info._transformed_ops[r.op]
      for output in copied_op._outputs:
        if output.name.endswith(r.name):
          boundary.append(output)
    assert len(boundary) == len(ts)
    substitute_backprops = [d_checkpoints[r] for r in ts]
    dv = tf_gradients(tf_gradient_function, boundary,
                      checkpoints_disconnected_other + xs,
                      substitute_backprops, "gradients",
                      _colocate_gradients_with_ops, gate_gradients,
                      aggregation_method, stop_gradients,
                      **kwargs)

    inputs_to_do_before = [d_checkpoints[r].op for r in ts if d_checkpoints[r] is not None]
    wait_to_do_ops = list(copied_ops) + [g.op for g in dv if g is not None]
    add_control_inputs_for_recomputation(wait_to_do_ops, inputs_to_do_before)

    # partial derivatives to the checkpointed nodes
    for r, dr in zip(checkpoints_other, dv[:len(checkpoints_other)]):
      if dr is not None:
        if d_checkpoints[r] is None:
          d_checkpoints[r] = dr
        else:
          d_checkpoints[r] += dr

    def _unsparsify(x):  # pylint: disable=missing-docstring
      if not isinstance(x, ops.IndexedSlices):
        return x
      assert x.dense_shape is not None, "GC Encountered sparse gradients of unknown shape"
      indices = x.indices
      while indices.shape.ndims < x.values.shape.ndims:

        indices = array_ops.expand_dims(indices, -1)
      if indices.dtype != x.dense_shape.dtype:
        shape = math_ops.cast(x.dense_shape, indices.dtype)
      else:
        shape = x.dense_shape
      return array_ops.scatter_nd(indices, x.values, shape)

    # partial derivatives to xs (usually the params of the neural net)
    d_xs_new = dv[len(checkpoints_other):]

    def _sum_grads(grads_list, index, grads_to_add):
      # Use sum instead of add_n, as add_n has to wait all tensors to be ready.
      grads_list[index] = sum([_unsparsify(grads_list[index]), _unsparsify(grads_to_add)])

    for j in range(len(xs)):
      if d_xs_new[j] is not None:
        if d_xs[j] is None:
          d_xs[j] = d_xs_new[j]
        else:
          if c_inp_xs[j]:
            with ops.control_dependencies(c_inp_xs[j]):
              _sum_grads(d_xs, j, d_xs_new[j])
          else:
            _sum_grads(d_xs, j, d_xs_new[j])
          c_inp_xs[j] = [d_xs[j]]

  fix_control_dependencies(checkpoints)

  # This is a workaround for TensorArray op whose consumer is in while_loop
  for op in ops.get_default_graph().get_operations():
    if 'TensorArray' in op.type and \
      is_consumer_in_while_loop(op) and "clear_after_read" in op.node_def.attr:
      op._set_attr("clear_after_read", attr_value_pb2.AttrValue(b=False))
      tf_logging.warn("set attr clear_after_read for {}, type {}" \
                  .format(op.name, op.type))
  if Env.get().config.gradient_checkpoint.check_gradients:
    assert len(base_gradients) == len(d_xs), \
        "GC grads {} vs base grads {}, not equal" \
        .format(len(d_xs), len(base_gradients))
    for bg, g in list(zip(base_gradients, d_xs)):
      if bg is None:
        assert g is None, "{} should be None".format(g)
      else:
        assert g is not None, "base {} vs gc {}".format(bg, g)
        if isinstance(bg, ops.IndexedSlices):
          _compare_two_tensor_shapes(bg.indices, g.indices)
          _compare_two_tensor_shapes(bg.values, g.values)
        else:
          _compare_two_tensor_shapes(bg, g)
          if not _compare_two_op_types(bg.op, g.op):
            raise ValueError("base type {} vs gc type {}".format(bg.op.type, g.op.type))

  return d_xs


def _compare_two_op_types(o1, o2):
  """Compare the type of two operations."""
  if o1.type == o2.type:
    return True
  # The results of add_n and sum are same, but with different types.
  if o1.type in ["AddN", "Add"] and o2.type in ["AddN", "Add"]:
    return True
  if o1.type == 'Identity':
    return _compare_two_op_types(o1.inputs[0].op, o2)
  if o2.type == 'Identity':
    return _compare_two_op_types(o1, o2.inputs[0].op)
  return False


def _compare_two_tensor_shapes(t1, t2):
  """Compare tensor shapes."""
  if t1.shape.as_list() != t2.shape.as_list():
    raise RuntimeError("Compare shape fail: base {} {} vs gc {} {}".format(t1.name, t1.shape.as_list(), t2.name, t2.shape.as_list()))
  return True


def add_control_inputs_for_recomputation(wait_to_do_ops, inputs_to_do_before):
  # To reduce the number of control dependencies,
  # add contral dependencies to entry topo only.
  sorted_ops = toposort_ops(wait_to_do_ops)
  for op in sorted_ops[0]:
    ci = [
        i for i in inputs_to_do_before
        if op.control_inputs is None or i not in op.control_inputs
    ]
    ge.add_control_inputs(op, ci)


def fix_control_dependencies(checkpoints):
  """Fix control dependency for checkpoint op."""
  all_ops = ops.get_default_graph().get_operations()
  grads = [op for op in all_ops if op.name.startswith("gradients")]
  checkpoints_ops = set(c.op for c in checkpoints)
  for grad in grads:
    if not grad.control_inputs:
      continue
    for ci in grad.control_inputs:
      for checkpoint_op in checkpoints_ops:
        if ci.name.endswith(checkpoint_op.name):
          ge.remove_control_inputs(grad, [ci])
          ge.add_control_inputs(grad, checkpoint_op)
