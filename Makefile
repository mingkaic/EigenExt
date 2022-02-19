
TMP_COVFILE := /tmp/coverage.info
CCOVER := bazel coverage --config asan --action_env="ASAN_OPTIONS=detect_leaks=0" --config gtest --config cc_coverage
COVERAGE_INFO_FILE := bazel-out/_coverage/_coverage_report.dat
COVFILE := coverage.info

.PHONY: clean_test_coverage
clean_test_coverage: ${COVERAGE_INFO_FILE}
	lcov --remove ${COVERAGE_INFO_FILE} '**/test/*' '**/mock/*' '**/*.pb.*' -o ${TMP_COVFILE}

.PHONY: cover
cover:
	${CCOVER} --instrumentation_filter 'EigenExt/*' //EigenExt:test
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o coverage.info

.PHONY: genhtml
genhtml: cover
	genhtml -o html $(COVFILE)

