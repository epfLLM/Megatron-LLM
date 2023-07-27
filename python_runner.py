import os
import tempfile
import warnings
from typing import List

warnings.filterwarnings("error")

# https://secure.idiap.ch/intranet/system/computing/ComputationGrid#computation-grid
gpu_configs = [
    "q_very_short_gpu", # <= 15 minutes
    "q_short_gpu",  # <= 3 hours
    "q_gpu",  # <= 48 hours
    "q_long_gpu"  # <= 120 hours
]

project_dir = f"/idiap/user/kmatoba/vit-small"
localenv_home = project_dir


def jobname_to_command(jobname: str) -> str:
    if "train_ssl" == jobname:
        arg_str = "--batch_size 512"
        python_command = project_dir + f"/train_ssl.py " + arg_str
    elif jobname in ["flattune17711"]:
        # arg_str = " " + "++prng.seed=17708 ++flattune.layernorm_lipschitz_weight=0.0"
        arg_str = " " + "++prng.seed=17711 ++flattune.layernorm_lipschitz_weight=0.0 ++flattune.diff_weight=1e4"
        python_command = project_dir + f"/flattune.py " + arg_str
    elif jobname in ["flattune17712"]:
        # arg_str = " " + "++prng.seed=17708 ++flattune.layernorm_lipschitz_weight=0.0"
        arg_str = " " + "++prng.seed=17712 ++flattune.layernorm_lipschitz_weight=1e-7 ++flattune.diff_weight=1e4"
        python_command = project_dir + f"/flattune.py " + arg_str
    elif jobname in ["flattune17710_exp"]:
        arg_str = " " + "++prng.seed=17710 ++flattune.layernorm_lipschitz_weight=0.10 ++flattune.diff_weight=1e5 " + \
                  "++flattune.interp_schedule_name=exponential ++flattune.interp_schedule_temperature=1.50 " + \
                  "++flattune.optimizer_name=SGD ++flattune.terminal_alpha=.15 ++base.tilde=/idiap/user/kmatoba"
        python_command = project_dir + f"/flattune.py " + arg_str
    else:
        raise ValueError(f"not found {jobname}")
    return python_command


def run_command_as_job(jobname: str,
                       command: str):
    to_prepend = f"PYTHONPATH={localenv_home}/ "
    # gpu_config = "q_very_short_gpu"
    # gpu_config = "q_short_gpu"
    gpu_config = "q_gpu"
    outfile_dir = project_dir + "/logs/"

    os.makedirs(outfile_dir, exist_ok=True)
    out_filename = f"{outfile_dir}{jobname}.out.txt"
    err_filename = f"{outfile_dir}{jobname}.err.txt"
    file_headers = ["#!/bin/bash",
                    f"#-N {jobname}",
                    f"#$-o {out_filename}",
                    f"#$-e {err_filename}"]
    # hosts = "vgn[efghi]*"
    # hosts = "vgn[e]*"  # https://secure.idiap.ch/intranet/system/computing/hosts-performances
    hosts = "*"
    project = "oh-ff"  # qconf -sprjl
    qsub = f"qsub -l {gpu_config}=TRUE -P {project} -w e -l pytorch -l 'h={hosts}' -V"
    python3 = f"{localenv_home}/localenv/bin/python3 -u"

    command_line = to_prepend + python3 + " " + command
    filelines = file_headers + [command_line]
    filecontents = ("#" * 80 + "\n") + \
                   "\n".join(filelines) + \
                   ("\n" + "#" * 80)

    filename = f"{jobname}.sh"
    tdir = tempfile.mkdtemp()
    path = os.path.join(tdir, filename)

    with open(path, 'w') as f:
        f.write(filecontents)

    print(f"File contents: \n{filecontents}")
    full_command = qsub + " " + path
    os.system(full_command)
    print(f"Running '{full_command}'")

    assert os.path.exists(path)
    os.unlink(path)
    os.rmdir(tdir)
    assert not os.path.exists(path)
    print(f"See output at '{out_filename}'")


def run_job(jobname: str) -> None:
    command = jobname_to_command(jobname)
    run_command_as_job(jobname, command)


if __name__ == "__main__":
    jobnames = [
        "flattune17710_exp"
    ]

    for idx, jobname in enumerate(jobnames):
        print(jobname)
        run_job(jobname)

    """
    qstat -u kmatoba -xml | tr '\n' ' ' | sed 's#<job_list[^>]*>#\n#g' | sed 's#<[^>]*>##g' | grep " " | column -t
    """
    # https://secure.idiap.ch/intranet/system/storage/generality
    # quota -s -Q -w
