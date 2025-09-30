import subprocess
from pathlib import Path

import ipdb
from tqdm import tqdm

if __name__ == "__main__":
    checkpoint_dirs = [
        Path("/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/verl/verl-projects/vlm_rl/checkpoints/mixture1_2-kl0.001-ent0/global_step_100"),
    ]

    for checkpoint_dir in tqdm(checkpoint_dirs):
        hf_checkpoint_dir = checkpoint_dir.parent / f"hf_{checkpoint_dir.name}"

        if hf_checkpoint_dir.exists():
            print(f"{hf_checkpoint_dir} already exists. Skipping...")
        else:
            command = (f"python -m verl.model_merger merge --backend=fsdp "
                       f"--local_dir={str(checkpoint_dir / 'actor')} --target_dir {str(hf_checkpoint_dir)}")
            subprocess.run(command, shell=True)