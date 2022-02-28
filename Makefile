PYTHON ?= python
ADDITIONAL_DEPS ?=

.PHONY: build
build: $(LIB)
	python setup.py bdist_wheel
	@printf "\033[0;32mPIP package built\033[0m: "
	@ls dist/*.whl

.PHONY: test
test: $(LIB)
	cd tests; $(MAKE) test

.PHONY: lint
lint:
	pip install lazy-object-proxy==1.6.0 pylint==1.9.4
	cd cc; $(MAKE) lint
	@$(PYTHON) -m pylint \
		--rcfile=.pylintrc --output-format=parseable --jobs=8 \
		$(shell find python/epl/ -type f -name '*.py') \
		$(shell find tests/ -type f -name '*.py') \
		$(shell find . -type f -name 'setup.py')

.PHONY: clean
clean:
	python setup.py clean

.PHONY: rebuild
rebuild:
	$(MAKE) clean
	$(MAKE) build

.PHONY: format
format:
	cd cc; $(MAKE) format
	find ./ -name "*.py" | xargs -n1 yapf -i --style='{based_on_style: pep8, indent_width: 2}'

.DEFAULT_GOAL := build
