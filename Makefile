
TMP_COVFILE := /tmp/coverage.info
CCOVER := bazel coverage --config asan --action_env="ASAN_OPTIONS=detect_leaks=0" --config gtest --config cc_coverage
COVERAGE_INFO_FILE := bazel-out/_coverage/_coverage_report.dat
COVFILE := coverage.info

.PHONY: clean_test_coverage
clean_test_coverage: ${COVERAGE_INFO_FILE}
	lcov --remove ${COVERAGE_INFO_FILE} '**/test/*' '**/mock/*' '**/*.pb.*' -o ${TMP_COVFILE}

coverage.info:
	${CCOVER} --instrumentation_filter 'EigenExt/*' //EigenExt:test
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o coverage.info

.PHONY: genhtml
genhtml: cover
	genhtml -o html $(COVFILE)

VERSION := $(shell cat VERSION)

.PHONY: conan_remote
conan_remote:
	conan remote add mingkaic-co "https://gitlab.com/api/v4/projects/23299689/packages/conan"

build/conanbuildinfo.cmake:
	conan install -if build .

.PHONY: conan_install
conan_install: build/conanbuildinfo.cmake

.PHONY: conan_build
conan_build: build/conanbuildinfo.cmake
	conan build -bf build .

.PHONY: conan_create
conan_create:
	conan create . mingkaic-co/stable

.PHONY: conan_upload
conan_upload:
	conan upload cisab/${VERSION}@mingkaic-co/stable --all --remote mingkaic-co

.PHONY: conan_create_n_upload
conan_create_n_upload: conan_install conan_create conan_upload

#### compile db (uncomment to generate)

#EXEC_ROOT := $(shell bazel info execution_root)

#COMP_FILE := $(shell bazel info bazel-bin)/compile_commands.json

#.PHONY: compdb
#compdb:
	#bazel build //:compdb
	#sed -i.bak "s@__EXEC_ROOT__@${EXEC_ROOT}@" "${COMP_FILE}"
	#ln -s "${COMP_FILE}" compile_commands.json
