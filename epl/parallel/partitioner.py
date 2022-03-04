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
"""Graph partitioner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import defaultdict
from tensorflow.python.platform import tf_logging

def partition_buckets(weights, bucket_size, num_stages):
  """partition weights to buckets given bucket_size"""
  # (start_index, total_per_bucket)
  buckets = [(0, 0)]
  for i, weight in enumerate(weights):
    cur_start, cur_total = buckets[-1]
    if cur_total + weight > bucket_size:
      if cur_total == 0:
        buckets[-1] = (cur_start, weight)
      else:
        buckets.append((i, weight))
      if len(buckets) > num_stages:
        return None
    else:
      buckets[-1] = (cur_start, cur_total + weight)
  return buckets


def partition_balance(operations, weights, num_stages):
  """partition operations based on weight."""
  if num_stages == 1:
    return [operations]
  # Norm weights to range [0, 100], in order to decrease the
  # search space.
  minw = min(weights)
  maxw = max(weights)
  if maxw - minw > 100:
    weights = [int((w - minw) / (maxw - minw) * 100) for w in weights]
  total_weights = int(sum(weights))
  min_bucket = total_weights // num_stages
  buckets = None
  stage_ops = []
  for bucket_size in range(min_bucket, total_weights):
    buckets = partition_buckets(weights, bucket_size, num_stages)
    if buckets:
      break
  if not buckets:
    return [operations]
  # append a right bound
  buckets.append((len(operations), -1))
  for i in range(len(buckets) - 1):
    stage = [operations[j] for j in range(buckets[i][0], buckets[i + 1][0])]
    stage_ops.append(stage)
  return stage_ops


def get_strategy_counter_key(operations, min_type_num=5):
  """Use the count of op types to find similar ops."""
  count_types = Counter(o.type for o in operations)
  if len(count_types) <= min_type_num: return None
  return str(sorted(count_types.items()))


def find_repeated_blocks(operations, max_depth=20, min_dup=4, min_ops=20):
  """Find repeated model blocks."""
  depth2scopes = defaultdict(dict)
  real_max_depth = 0
  for depth in range(1, max_depth):
    if depth > 1 and depth > real_max_depth:
      break
    for op in operations:
      scopes = op.name.split("/")
      if depth == 1:
        real_max_depth = max(real_max_depth, len(scopes))
      if len(scopes) < depth: continue
      scope_name = "/".join(scopes[:depth])
      scope_ops = depth2scopes[depth].get(scope_name, [])
      scope_ops.append(op)
      depth2scopes[depth][scope_name] = scope_ops
  candidates = []
  visited_scopes = set()
  for depth, scope2ops in depth2scopes.items():
    count2ops = defaultdict(list)
    all_visited = True
    for scope, vs in scope2ops.items():
      if visited_scopes and any(scope.startswith(x) for x in visited_scopes):
        continue
      all_visited = False
      key = get_strategy_counter_key(vs)
      if key:
        count2ops[key].append((scope, vs))
    if all_visited:
      break
    for key, similar_ops in count2ops.items():
      if len(similar_ops) >= min_dup:
        tf_logging.info("Find {} repeated blocks with {} ops" \
            .format(len(similar_ops), len(similar_ops[0][1])))
      if len(similar_ops) >= min_dup and len(similar_ops[0][1]) > min_ops:
        candidates.extend([operations for _, operations in similar_ops])
        for scope, block in similar_ops:
          visited_scopes.add(scope)
          tf_logging.info("Find scope {}, block op number: {}" \
                      .format(scope, len(block)))
  # sort by op name
  candidates = sorted(candidates, key=lambda blocks: blocks[0].name)
  return candidates


def partition_stages(operations, weights, num_stages, enable_logging=False):
  """partition operations based on weight, the partitioned size == num_stages"""
  if num_stages <= 0:
    raise ValueError("partition_stages requires num_stages>=1, got {}" \
        .format(num_stages))
  if num_stages == 1:
    return [operations]
  if len(operations) <= num_stages:
    final_buckets = [[op] for op in operations]
    for _ in range(num_stages-len(operations)):
      final_buckets.append([])
    return final_buckets
  minw = min(weights)
  maxw = max(weights)
  norm_range = 100
  if maxw - minw > norm_range:
    weights = [(w - minw) / (maxw - minw) * norm_range for w in weights]
  weights = [max(1, int(w)) for w in weights]

  total_weights = sum(weights)
  min_bucket = int(total_weights / num_stages)
  final_buckets = None
  final_weights = None
  for bucket_size in range(min_bucket, 0, -1):
    buckets = [[]]
    bucket_weights = [0]
    for op, w in list(zip(operations, weights)):
      if bucket_weights[-1] >= bucket_size and len(buckets) < num_stages:
        bucket_weights.append(0)
        buckets.append([])
      bucket_weights[-1] += w
      buckets[-1].append(op)
    if len(buckets) == num_stages:
      final_buckets = buckets
      final_weights = bucket_weights
      break
  if enable_logging:
    tf_logging.info("Parititioned {} items into {} stages, with partition " \
                    "weights {}".format(len(operations), \
                                        num_stages, final_weights))
  return final_buckets
