PYTHON ?= python

GPU_1 = 7
GPU_2 = 6,7
GPU_3 = 5,6,7
GPU_4 = 4,5,6,7
GPU_8 = 0,1,2,3,4,5,6,7

.PHONY: test
test:
	PYTHONPATH=../python/ python -m epl.utils.launcher --num_workers=2 --gpu_per_worker=1 --debug=True test_launcher.sh
	CUDA_VISIBLE_DEVICES='' ./launch.sh header_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_4) ./launch.sh strategy_new_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_4) ./launch.sh auto_cluster_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_2) ./launch.sh estimator_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_2) ./launch.sh estimator_loss_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_2) ./launch.sh optimizer_helper_test.py
	CUDA_VISIBLE_DEVICES='' ./launch.sh initializers_test.py
	CUDA_VISIBLE_DEVICES='' ./launch.sh phase_test.py
	CUDA_VISIBLE_DEVICES='' ./launch.sh strategy_test.py
	CUDA_VISIBLE_DEVICES='' ./launch.sh cluster_test.py
	CUDA_VISIBLE_DEVICES='' ./launch.sh cluster_test_with_aware.py
	CUDA_VISIBLE_DEVICES=$(GPU_4) ./launch.sh cluster_test_with_visible_devices.py
	CUDA_VISIBLE_DEVICES='' ./launch.sh utils_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_2) ./launch.sh communicator_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_4) ./launch.sh graph_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_4) ./launch.sh while_loop_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_8) ./launch.sh scheduler_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_8) ./launch.sh optimizer_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_4) ./launch.sh run_hook_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_3) ./launch.sh collections_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_4) ./launch.sh summary_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_4) ./launch.sh function_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh saver_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh env_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh planner_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh config_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh config_env_test.py export_config_env.sh
	CUDA_VISIBLE_DEVICES=$(GPU_4) ./launch.sh auto_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh gradient_checkpoint_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh hooks_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh shape_inference_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh auto_gradient_checkpoint_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh profiler_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_2) ./launch.sh set_default_scope_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh add_weights_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh split_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh zero_test.py
	CUDA_VISIBLE_DEVICES='' ./launch.sh shape_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh flops_hook_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh operation_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh memory_profiler_hook_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh offload_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_1) ./launch.sh amp_test.py
	CUDA_VISIBLE_DEVICES=$(GPU_2) ./launch.sh amp_parallel_test.py

.PHONY: clean
clean:
	@find -name '*.pyc' -exec rm -fr {} \;

.DEFAULT_GOAL := test
