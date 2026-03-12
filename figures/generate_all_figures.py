import subprocess
import yaml
import glob
from tqdm import tqdm

with open("config.yml") as f:
    config = yaml.safe_load(f)

notebooks = sorted(glob.glob("figures/*.ipynb"))

for nb in tqdm(notebooks):
    print(f"Running {nb}")
    subprocess.run([
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--output-dir", "executed_notebooks",
        nb
    ], check=True)