TFCOLL_WITH_CUDA ?= ON
TFCOLL_WITH_NCCL ?= ON
TFCOLL_USE_CXX11_ABI ?= 0

PYTHON ?= python
CXX ?= g++
LOCAL_HOME ?= /usr/local
PAI_HOME ?= /home/pai

CFLAGS := -O3 -g -DNDEBUG \
	-D_GLIBCXX_USE_CXX11_ABI=$(TFCOLL_USE_CXX11_ABI) \
	-DEIGEN_MPL2_ONLY \
	-DEIGEN_MAX_ALIGN_BYTES=64 \
	-DEIGEN_HAS_TYPE_TRAITS=0 \
	-I$(PAI_HOME)/include \
	-I. \
	-I./tensorflow_include/ \
	$(shell $(PYTHON) ../epl/utils/tf_cflags.py 2>/dev/null)

CXX_CFLAGS := \
	-std=c++11 \
	-fstack-protector \
	-Wall \
	-Werror \
	-Wno-sign-compare \
	-Wformat \
	-Wformat-security

LDFLAGS := -shared \
	-znoexecstack \
	-zrelro \
	-znow \
	-fstack-protector \
	-L$(PAI_HOME)/lib \
	-Wl,-rpath='$$ORIGIN:$$ORIGIN/../../tensorflow' \
	$(shell $(PYTHON) ../epl/utils/tf_ldflags.py 2>/dev/null)

ifeq ($(TFCOLL_WITH_CUDA),ON)
NVCC ?= nvcc
CUDA_HOME ?= $(LOCAL_HOME)/cuda
CFLAGS := $(CFLAGS) \
	-DGOOGLE_CUDA=1 \
	-I$(CUDA_HOME)/include
LDFLAGS := $(LDFLAGS) \
	-L$(CUDA_HOME)/lib64 \
	-lcudart

ifeq ($(TFCOLL_WITH_NCCL),ON)
NCCL_HOME ?= $(LOCAL_HOME)/cuda
CFLAGS := $(CFLAGS) \
	-I$(NCCL_HOME)/include
LDFLAGS := $(LDFLAGS) \
	-L$(NCCL_HOME)/lib \
	-L$(NCCL_HOME)/lib64 \
	-lnccl
endif
endif

SOURCES := $(shell \
	find communicators/ -type f \
	\( -name "*.cc" ! -name "*.cu*" ! -name "*test*" ! -name "*benchmark*" \) \
	-exec realpath {} --relative-to . \;)

OBJS := $(SOURCES:.cc=.o)
$(OBJS): %.o:%.cc
	mkdir -p $(dir $@)
	$(CXX) $(CXX_CFLAGS) $(CFLAGS) \
	-MMD -MP -MF $<.d -o $@ -c $< -fPIC -fpic

ifeq ($(TFCOLL_WITH_CUDA),ON)
CU_SOURCES := $(shell \
	find communicators/ -type f \
	\( -name '*.cu.cc' ! -name "*test*" ! -name "*benchmark*" \) \
	-exec realpath {} --relative-to . \;)

CU_OBJS := $(CU_SOURCES:.cc=.o)
$(CU_OBJS): %.o:%.cc
	mkdir -p $(dir $@)
	@$(NVCC) -M $< $(CFLAGS) -x cu \
	 | grep -v '/usr/' \
	 | sed 's|$(notdir $@)|$@|g' \
	 | sed 's|\./||g' \
	 > $<.d
	$(NVCC) \
		--std=c++11 \
		--expt-relaxed-constexpr \
		--expt-extended-lambda \
		--disable-warnings \
		-o $@ -c $< $(CFLAGS) -x cu \
		-Xcompiler -fPIC -fpic
endif

DEPS := $(shell \
	find -type f -name "*.d" \
	-exec realpath {} --relative-to . \;)

-include $(DEPS)

LIB := ../epl/communicators/libcommunicators.so
$(LIB):
ifeq ($(TFCOLL_WITH_CUDA),ON)
$(LIB): $(OBJS) $(CU_OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 \
	-o $@ $^ $(LDFLAGS) -fPIC -fpic
else
$(LIB): $(OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 \
	-o $@ $^ $(LDFLAGS) -fPIC -fpic
endif

.PHONY: build
build: $(LIB)

.PHONY: lint
lint:
	./clang-lint.sh \
		$(shell find -type f -name '*.h') \
        	$(shell find -type f -name '*.cc')

.PHONY: format
format:
	@clang-format -i \
		$(shell find -type f -name '*.h') \
		$(shell find -type f -name '*.cc')

.PHONY: clean
clean:
	@rm -f $(LIB) $(OBJS) $(CU_OBJS) $(UTILS_LIB) $(UTILS_OBJS) $(DEPS)

.PHONY: rebuild
rebuild:
	$(MAKE) clean
	$(MAKE) build

.DEFAULT_GOAL := build
