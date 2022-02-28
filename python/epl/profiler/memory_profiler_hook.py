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
"""Profile peak memory."""
import inspect
import os
import pickle

import tensorflow
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.core.protobuf import config_pb2

import pandas as pd


def profile_memory(run_metadata, prefix_name="", visualize=True):
  """Return dictionary of memory usage (bytes) for each device."""
  dev_stats = run_metadata.step_stats.dev_stats
  results = []
  for dev_stat in dev_stats:
    res = _peak_from_nodestats(dev_stat.node_stats,
                               dev_stat.device,
                               prefix_name,
                               visualize)
    if res["peak_memory"] > 0:
      results.append(res)
      tf_logging.info("Device: {}, peak memory: {:.4f}GB" \
                      .format(dev_stat.device, res["peak_memory"] / (1024.0 ** 3)))
  return results


def _timeline_from_nodestats(nodestats, device_name):
  """Return sorted memory allocation records from list of nodestats."""
  if not nodestats:
    return []
  lines = []
  peak_mem = 0
  for node in nodestats:
    for mem in node.memory:
      try:
        records = mem.allocation_records
      except:  # pylint: disable=bare-except
        records = []
      allocator = mem.allocator_name
      peak_mem = max(peak_mem, mem.allocator_bytes_in_use)
      if records:
        for record in records:
          data = {
              "time": record.alloc_micros,
              "name": node.node_name,
              "allocated_bytes": record.alloc_bytes,
              "allocator_type": allocator,
              "persist_size": node.memory_stats.persistent_memory_size
          }
          lines.append(data)
      else:
        data = {
            "time": node.all_start_micros,
            "name": node.node_name,
            "allocated_bytes": node.memory_stats.persistent_memory_size,
            "allocator_type": allocator,
            "persist_size": node.memory_stats.persistent_memory_size
        }
        lines.append(data)
    if (not node.memory) and node.memory_stats.persistent_memory_size:
      data = {
          "time": node.all_start_micros,
          "name": node.node_name,
          "allocated_bytes": node.memory_stats.persistent_memory_size,
          "allocator_type": device_name,
          "persist_size": node.memory_stats.persistent_memory_size,
          "allocator_bytes_in_use": -1
      }
      lines.append(data)
  return sorted(lines, key=lambda x: x["time"]), peak_mem


def _mark_phase(df,
                ax,
                keywords,
                name,
                color1,
                color2,
                match_fn=None,
                match_type="start"):
  """Mark phase with colors."""
  alpha = 1.0
  if match_fn is None:
    if match_type == "start":
      match_fn = lambda name: any(name.lower().startswith(k) for k in keywords)
    elif match_type == "in":
      match_fn = lambda name: any(k in name.lower() for k in keywords)
  df[name + "_allocate"] = df.apply(
      lambda x: match_fn(x["name"]) and x["allocated_bytes"] > 0, axis=1)
  df[name + "_release"] = df.apply(
      lambda x: match_fn(x["name"]) and x["allocated_bytes"] < 0, axis=1)
  if any(df[name + "_allocate"]):
    ax.fill_between(df.index,
                    df["cumsum"],
                    facecolor=color1,
                    alpha=alpha,
                    label=name + "_allocate",
                    hatch="",
                    where=df[name + "_allocate"])
  if any(df[name + "_release"]):
    ax.fill_between(df.index,
                    df["cumsum"],
                    facecolor=color2,
                    alpha=alpha,
                    hatch="",
                    label=name + "_release",
                    where=df[name + "_release"])
  return df


def get_func_default_args(func):
  """Get default args of func."""
  args, _, _, defaults = inspect.getargspec(func)
  if not defaults:
    return {}
  return dict(zip(args[-len(defaults):], defaults))


def get_optimizer_prefix():
  """Get optimizer names."""
  optimizer_cls = [o for n, o in tensorflow.train.__dict__.items() \
                   if 'Optimizer' in n]
  prefix = []
  for cls in optimizer_cls:
    name = get_func_default_args(cls.__init__).get('name')
    if name:
      prefix.append(name)
  return prefix


def mark_phase(df, ax):
  """mark phase"""
  alpha = 1.0
  df["persist"] = df.apply(lambda x: x["persist_size"] > 0, axis=1)
  ax.fill_between(df.index,
                  df["cumsum"],
                  facecolor="orange",
                  alpha=alpha,
                  label="persist",
                  hatch="x",
                  where=df["persist"])
  _mark_phase(df, ax, ["gradients"], "gradients", "blue", "yellow")
  _mark_phase(df, ax, get_optimizer_prefix(), "optimizer", "red", "green")


def _simplify_device_name(device_name):
  if not device_name: return "unknown_device"
  return device_name.replace("/", "_").replace(":", "-")


def _peak_from_nodestats(nodestats, device_name=None, prefix_name="",
                         visualize=True):
  """Given a list of NodeExecStats messages, construct memory timeline."""
  timeline, peak_memory_bytes = _timeline_from_nodestats(nodestats, device_name)
  timeline = [t for t in timeline if t["allocated_bytes"] != 0]
  stats = {'device': device_name}
  stats['persist'] = sum([i['persist_size'] for i in timeline])
  peak = total_memory = 0
  for record in timeline:
    total_memory += int(record["allocated_bytes"])
    peak = max(total_memory, peak)
  stats['peak_memory'] = max(peak, peak_memory_bytes)

  if timeline and visualize:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    df = pd.DataFrame(timeline)
    save_file = "{}_{}.csv".format(prefix_name,
                                   _simplify_device_name(device_name))
    df["cumsum"] = df["allocated_bytes"].cumsum()
    with gfile.Open(save_file, "wb") as f:
      df.to_csv(f, index=False)
    df = df.reset_index()
    ax = df.plot(x="index", y="cumsum", figsize=(15, 5))
    mark_phase(df, ax)
    ax.legend(loc="upper right")
    plt.title(device_name)
    save_name = os.path.join(os.path.dirname(save_file),
                             os.path.basename(save_file) + ".png")
    tf_logging.info("Save to memory peak figure to {}".format(save_name))
    with gfile.Open(save_name, "wb") as f:
      ax.get_figure().savefig(f)
  return stats


class MemoryProfilerHook(session_run_hook.SessionRunHook):
  """Captures memory profiling information every N steps."""
  def __init__(self, save_steps=1, max_steps=5, output_dir="", visualize=True,
               dump_metadata=False):
    super(MemoryProfilerHook, self).__init__()

    if save_steps > max_steps:
      raise ValueError("save_steps {} should <= max_steps {}" \
                       "to make profiling work.".format(save_steps, max_steps))
    self.output_dir = output_dir
    self._timer = SecondOrStepTimer(every_steps=save_steps)
    self.max_steps = max_steps
    self.stop = False
    self.visualize = visualize
    self.dump_metadata = dump_metadata

  def begin(self):
    """Called once before using the session."""
    self._next_step = None
    # pylint: disable=protected-access
    self._global_step_tensor = training_util._get_or_create_global_step_read()
    assert self._global_step_tensor is not None, \
        "Global step should be created to use MemoryProfilerHook."
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

  def before_run(self, run_context):  # pylint: disable=unused-argument
    """Called before each call to run()."""
    self._request_summary = (not self.stop
                             and self._next_step is not None
                             and self._timer.should_trigger_for_step(
                                 self._next_step))
    requests = {"global_step": self._global_step_tensor}
    opts = (config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
            if self._request_summary else None)

    return SessionRunArgs(requests, options=opts)

  def after_run(self, run_context, run_values):
    """Called after each call to run()."""
    stale_global_step = run_values.results["global_step"]
    if self._next_step is None:
      self._timer.update_last_triggered_step(stale_global_step)
    global_step = stale_global_step + 1
    if self._request_summary:
      global_step = run_context.session.run(self._global_step_tensor)
      self._timer.update_last_triggered_step(global_step)
      self._save(global_step, run_values.run_metadata)
      if global_step >= self.max_steps:
        self.stop = True
    self._next_step = global_step + 1

  def _save(self, step, run_metadata):
    """Save profile results."""
    prefix_name = os.path.join(self.output_dir,
                               "profile_step_{}".format(step))
    if self.dump_metadata:
      tf_logging.info("Save memory profiling result of step %d into dir '%s'.",
                      step, self.output_dir)
      bin_file = os.path.join(self.output_dir,
                              "run_metadata_{}.bin".format(step))
      with gfile.Open(bin_file, "wb") as f:
        pickle.dump(run_metadata, f)
    profile_memory(run_metadata, prefix_name, self.visualize)
