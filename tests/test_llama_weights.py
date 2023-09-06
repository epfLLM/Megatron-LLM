import re
import pytest
import shutil
from pathlib import Path
from typing import Optional, Iterator
from tempfile import TemporaryDirectory
from subprocess import PIPE, Popen


# ===
# = Arguments
# ===
@pytest.fixture(scope="session", params=[1, 2])
def llama_version(request) -> int:
    return request.param


@pytest.fixture(scope="session")
def llama_meta(pytestconfig, llama_version: int ) -> Path:
    if llama_version == 1:
        return pytestconfig.getoption("llama_path")
    return pytestconfig.getoption("llama2_path")


@pytest.fixture(scope="session")
def cache_dir(pytestconfig) -> Optional[Path]:
    return pytestconfig.getoption("cache_path")


@pytest.fixture(scope="session")
def data(pytestconfig) -> Path:
    return pytestconfig.getoption("data_path")


@pytest.fixture(scope="session")
def vocab(pytestconfig) -> Path:
    return pytestconfig.getoption("vocab_path")

@pytest.fixture(scope="session")
def root_dir(pytestconfig) -> TemporaryDirectory:
    prefix = pytestconfig.getoption("tmp_dir")
    prefix = None if prefix is None else str(prefix/"tmp")
    return TemporaryDirectory(prefix=prefix)


# ===
# = Paths
# ===
@pytest.fixture(scope="session")
def root(root_dir, llama_version: int) -> Path:
    return Path(f"{root_dir.name}-{llama_version}")

@pytest.fixture(scope="session")
def llama_meta2mega(root: Path) -> Path:
    return root/"llama-meta2mega"

@pytest.fixture(scope="session")
def llama_hf2mega(root: Path) -> Path:
    return root/"llama-hf2mega"

@pytest.fixture(scope="session")
def vocab_hf2mega(llama_hf2mega: Path) -> Path:
    return llama_hf2mega/"tokenizer.model"

@pytest.fixture(scope="session")
def llama_sharded(root: Path) -> Path:
    return root/"llama-sharded"

@pytest.fixture(scope="session")
def llama_unsharded(root: Path) -> Path:
    return root/"llama-unsharded"

@pytest.fixture(scope="session")
def llama_mega2hf(root: Path) -> Path:
    return root/"llama-mega2hf"

@pytest.fixture(scope="session")
def llama_unsharded2hf(root: Path) -> Path:
    return root/"llama-unsharded2hf"


# ===
# = Utils
# ===
def execute(cmd: list[str]) -> Iterator[str]:
    with Popen(cmd, stdout=PIPE, text=True) as proc:
        yield from map(lambda line: line.strip(), iter(proc.stdout.readline, ""))
        assert proc.wait() == 0


def verify_correctness(our_path: Path, cache_dir: Optional[Path], data: Path,
                       vocab: Path, llama_v: int = 2) -> list[float]:
    llama_version = llama_v
    model_name = "llama" if llama_version == 1 else "llama2"
    distributed_args = ["--nproc_per_node=1", "--nnodes=1",
                        "--node_rank=0", "--master_addr=localhost",
                        "--master_port=8001"]
    main_args = [f"--model_name={model_name}", f"--load={our_path}",
                 f"--data_path={data}", "--no_new_tokens",
                 "--tokenizer_type=SentencePieceTokenizer",
                 "--model_size=7", f"--vocab_file={vocab}"]
    extra_args = ["--hidden_dropout=0.0", "--attention_dropout=0.0", 
                  "--no_bias_dropout_fusion", "--no_bias_gelu_fusion"]
    cmd = ["torchrun"] + distributed_args + ["verify_correctness.py"] \
                       + main_args + extra_args
    if cache_dir is not None:
        cmd.append(f"--huggingface_cache={cache_dir}")
    if llama_version == 1:
        cmd.append("--layernorm_epsilon=1e-6")

    max_errors = []
    for line in execute(cmd):
        if any(key in line for key in ["Iteration", "Max abs", "Abs loss"]):
            print(line)
        if rmatch := re.match(fr"^.*max=([0-9]+\.[0-9]+).*$", line):
            max_errors.append(float(rmatch.group(1)))
    assert sum(max_errors)/len(max_errors) <= 0.001, "Avg max error exceeds tolerance (0.001)"
    return max_errors


def shard(load_dir: Path, save_dir: Path, llama_v: int = 2, tp: int = 1, pp: int = 1):
    llama_version = llama_v
    model_type = "llama" if llama_version == 1 else "llama2"
    cmd = ["python", "tools/checkpoint_util.py", f"--load_dir={load_dir}",
           f"--save_dir={save_dir}", f"--model_type={model_type}", "--true_vocab_size=32000",
           f"--target_tensor_parallel_size={tp}", f"--target_pipeline_parallel_size={pp}"]
    ignores = {"---", "...", "Setting"}
    for line in execute(cmd):
        if all(avoid not in line for avoid in ignores):
            print(line)


def mega2hf(load_dir: Path, out_dir: Path, llama_v: int):
    model_type = "llama" if llama_v == 1 else "llama2"
    with Popen(["python", "weights_conversion/megatron_to_hf.py", f"--model={model_type}",
                f"--input_dir={load_dir}", f"--output_dir={out_dir}"]) as proc:
        assert proc.wait() == 0


# ===
# = Tests
# ===
@pytest.mark.incremental
class TestLlamaWeights:
    def test_path_exists(self, llama_meta: Path):
        assert llama_meta.exists() and llama_meta.is_dir()

    def test_meta2mega(self, llama_meta2mega: Path, llama_meta: Path,
                       llama_version: int,
                       cache_dir: Optional[Path], data: Path, vocab: Path):
        assert not llama_meta2mega.exists()
        model_name = "llama" if llama_version == 1 else "llama2"
        with Popen(["python", Path("weights_conversion")/"hf_to_megatron.py",
                    model_name, "--size=7", f"--out={llama_meta2mega}",
                    f"--cache-dir={llama_meta}"]) as proc:
            assert proc.wait() == 0
        assert llama_meta2mega.exists()
        if llama_version == 1:
            llama_meta = cache_dir
        verify_correctness(llama_meta2mega, llama_meta, data, vocab, llama_v=llama_version)
        shutil.rmtree(llama_meta2mega)  # all future tests will only use llama_hf2mega

    def test_hf2mega(self, llama_hf2mega: Path, cache_dir: Optional[Path],
                     data: Path, vocab_hf2mega: Path, llama_version: int):
        assert not llama_hf2mega.exists()
        model_name = "llama" if llama_version == 1 else "llama2"
        cmd = ["python", Path("weights_conversion")/"hf_to_megatron.py",
               model_name, "--size=7", f"--out={llama_hf2mega}"]
        if cache_dir is not None:
            cmd.append(f"--cache-dir={cache_dir}")
        with Popen(cmd) as proc:
            assert proc.wait() == 0
        assert llama_hf2mega.exists()
        verify_correctness(llama_hf2mega, cache_dir, data, vocab_hf2mega, llama_v=llama_version)

    def test_metallama_verification(self, llama_hf2mega: Path, llama_meta: Path,
                                    llama_version: int, data: Path, vocab: Path):
        verify_correctness(llama_hf2mega, llama_meta, data, vocab, llama_v=llama_version)

    def test_shard_unshard(self, llama_hf2mega: Path, llama_sharded: Path,
                           llama_unsharded: Path, cache_dir: Optional[Path],
                           llama_version: int, data: Path, vocab_hf2mega: Path):
        print("sharding to tp=2, pp=2")
        shard(llama_hf2mega, llama_sharded, llama_v=llama_version, tp=2, pp=2)
        assert llama_sharded.exists()
        print("merging back to tp=1, pp=1")
        shard(llama_sharded, llama_unsharded, llama_v=llama_version, tp=1, pp=1)
        assert llama_unsharded.exists()
        verify_correctness(llama_unsharded, cache_dir, data, vocab_hf2mega, llama_v=llama_version)

    def test_mega2hf(self, llama_hf2mega: Path, llama_mega2hf: Path,
                     cache_dir: Optional[Path], data: Path, vocab_hf2mega: Path,
                     llama_version: int ):
        mega2hf(llama_hf2mega, llama_mega2hf, llama_version)
        verify_correctness(llama_mega2hf, cache_dir, data, vocab_hf2mega, llama_v=llama_version)

    def test_unsharded2hf(self, llama_unsharded: Path, llama_unsharded2hf: Path,
                          cache_dir: Optional[Path], data: Path, vocab_hf2mega: Path,
                          llama_version: int):
        mega2hf(llama_unsharded, llama_unsharded2hf, llama_version)
        verify_correctness(llama_unsharded2hf, cache_dir, data, vocab_hf2mega, llama_v=llama_version)
