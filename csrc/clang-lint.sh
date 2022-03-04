#!/bin/bash

set -x

num_replacements=$(clang-format -style=file -output-replacements-xml $@ | grep -c '<replacement ')

exit $num_replacements
