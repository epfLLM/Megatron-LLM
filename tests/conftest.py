import pytest
from pathlib import Path


_test_failed_incremental: dict[str, dict[tuple[int, ...], str]] = {}


def pytest_addoption(parser):
    parser.addoption("--cache_path", type=Path,
                     help="Huggingface cache path (optional)")
    parser.addoption("--llama_path", type=Path, required=True,
                     help="Path where the raw 7B weights are located (llama1)")
    parser.addoption("--llama2_path", type=Path, required=True,
                     help="Path where the raw llama-2-7b weights are located")
    parser.addoption("--tmp_dir", type=Path,
                     help="Prefix of the tempdir to create (optional)")
    parser.addoption("--data_path", type=Path, required=True,
                     help="Path where the megatron dataset is located")
    parser.addoption("--vocab_path", type=Path, required=True,
                     help="Meta's vocabfile")


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        # incremental marker is used
        if call.excinfo is not None:
            # the test has failed
            # retrieve the class name of the test
            cls_name = str(item.cls)
            # retrieve the index of the test (if parametrize is used in combination with incremental)
            parametrize_index = (
                tuple(item.callspec.indices.values())
                if hasattr(item, "callspec")
                else ()
            )
            # retrieve the name of the test function
            test_name = item.originalname or item.name
            # store in _test_failed_incremental the original name of the failed test
            _test_failed_incremental.setdefault(cls_name, {}).setdefault(
                parametrize_index, test_name
            )


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        # retrieve the class name of the test
        cls_name = str(item.cls)
        # check if a previous test has failed for this class
        if cls_name in _test_failed_incremental:
            # retrieve the index of the test (if parametrize is used in combination with incremental)
            parametrize_index = (
                tuple(item.callspec.indices.values())
                if hasattr(item, "callspec")
                else ()
            )
            # retrieve the name of the first test function to fail for this class name and index
            test_name = _test_failed_incremental[cls_name].get(parametrize_index, None)
            # if name found, test has failed for the combination of class name & test name
            if test_name is not None:
                pytest.xfail("previous test failed ({})".format(test_name))
