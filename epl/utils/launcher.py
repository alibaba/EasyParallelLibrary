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
"""EPL launcher."""

import argparse
from datetime import date
import os
import subprocess
from socket import socket
import time


def launch_command_parser():
  """Create launch command parser."""

  parser = argparse.ArgumentParser("EPL launcher command")
  parser.add_argument(
      "--num_workers",
      type=int,
      help="The total number of workers used in this training.",
      required=True
  )
  parser.add_argument(
      "--gpu_per_worker",
      type=int,
      help="Number of gpu number used per worker.",
      required=True
  )
  parser.add_argument(
      "--machine_rank",
      type=int,
      default=None,
      help="The rank of the machine on which this script is launched.",
  )
  parser.add_argument(
      "--machine_list",
      type=str,
      default=None,
      help="A list of machines in the format of 'ip:port', seperated by ','.",
  )
  parser.add_argument(
      "training_script",
      type=str,
      help="the training script path to be launched in parallel",
  )
  parser.add_argument(
      "--task_label",
      type=str,
      help="Label to describe and indentify different training tasks.",
      default=None
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      help="Directory to save log.",
      default=None
  )
  parser.add_argument(
      "--debug",
      type=bool,
      default=False,
      help="debug mode, make it easy for python pdb."
  )
  return parser


def execute_cmd(cmd):
  """Execute commands."""
  return subprocess.call(cmd, shell=True)


def get_gpu_memory():
  """Get free gpu memory ratio."""
  command = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
  memory_free_values = [float(x.split()[0]) for _, x in enumerate(memory_free_info)]
  command = "nvidia-smi --query-gpu=memory.total --format=csv"
  memory_total_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
  memory_total_values = [float(x.split()[0]) for _, x in enumerate(memory_total_info)]
  memory_ratios = [free / total for free, total in list(zip(memory_free_values, memory_total_values))]
  return memory_ratios


def get_free_port():
  """Get a free port."""
  s = socket()
  s.bind(('', 0))
  return s.getsockname()[1]


def get_tf_config(worker_num, index, ports, machine_list=None):
  """Generate tf config."""
  hosts = []
  if machine_list is not None:
    hosts = machine_list.split(",")
    hosts = [host if host.startswith('"') else '"{}"'.format(host) for host in hosts]
  else:
    ip = '127.0.0.1'
    for i in range(worker_num):
      port = ports[i]
      hosts.append('"{}:{}"'.format(ip, port))
  tf_config = 'TF_CONFIG=\'{"cluster":{"worker":[' + ','.join(hosts) +']},"task":{"type":"worker","index":' + str(index) + '}}\''
  return tf_config


def get_visible_devices(num_gpu=1, offset=0):
  devices = []
  for i in range(num_gpu):
    devices.append(i + offset)
  return ','.join(str(i) for i in devices)


def kill_process(entry_file):
  return 'pgrep -f "^python {}" | xargs -r -n1 kill -9 || true'.format(entry_file)


def get_run_script(args):
  """generate run script."""
  script = ''
  gpu_memories = get_gpu_memory()
  offset = 0
  for ind, gm in enumerate(gpu_memories):
    if gm >= 0.99:
      offset = ind
      break
  output_dir = args.log_dir
  if output_dir is None:
    output_dir = "experiments/{}_{}w{}g_{}".format(date.today(), args.num_workers, args.gpu_per_worker,
                                                   os.path.basename(args.training_script).split('.')[0])
    if args.task_label:
      output_dir += '_' + args.task_label + '/'

  ports = [get_free_port() for _ in range(args.num_workers)]
  for index in range(args.num_workers-1, -1, -1):
    if args.machine_rank != None and args.machine_rank != index:
      continue
    task_dir = os.path.join(output_dir, str(index))
    if not os.path.exists(task_dir):
      os.makedirs(task_dir)
    script += "export GPU_STATUS_FILE={}/GPU_STATUS_{}.json\n".format(task_dir, index)
    script += get_tf_config(args.num_workers, index, ports, args.machine_list)
    script += ' CUDA_VISIBLE_DEVICES=' + get_visible_devices(args.gpu_per_worker, offset)
    cmd = 'python' if args.training_script.endswith('.py') else 'bash'
    script += ' {} {}'.format(cmd, args.training_script) + ' ' + task_dir
    if args.debug and index == 0:
      print("Enable debug mode")
    else:
      log_file = '{}/stderr_{}.log'.format(task_dir, index)
      script += '> {} 2>&1&'.format(log_file)
      script += "\necho [task_index: {}] log file: {}".format(index, log_file)
    script += '\n'
    offset += args.gpu_per_worker
  return script


def run_script(args):
  """local script"""
  entry_file = args.training_script

  cmd = get_run_script(args)
  retry = 1
  script_header = "EPL Training Script"
  indent_len = 30
  print("\n{} {} {}".format("="*indent_len, script_header, "="*indent_len))
  print(cmd.strip())
  print("="*(indent_len*2+len(script_header)+2))
  print("\n")
  execute_cmd(kill_process(entry_file))
  while retry > 0:
    code = execute_cmd(cmd)
    execute_cmd(kill_process(entry_file))
    time.sleep(2)
    if code == 0:
      break
    retry -= 1
  return [code]


def multi_gpu_local_launcher(args):
  """Local launcher for multiple GPUs"""
  run_script(args)


def main():
  parser = launch_command_parser()
  args = parser.parse_args()
  multi_gpu_local_launcher(args)


if __name__ == "__main__":
  main()
