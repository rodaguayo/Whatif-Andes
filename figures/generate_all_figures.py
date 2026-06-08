import subprocess
import glob
import os
from tqdm import tqdm


from src.utils import load_config
config = load_config()

notebooks = sorted(glob.glob("figures/*.ipynb"))
figures = sorted([f for f in glob.glob("figures/*.svg") if "slide" not in f and "workflow" not in f])

# Ask user if they want to backup existing figures
backup = input("Rename existing figures to _old for comparison? (y/n): ").lower() == 'y'

if backup:
    for fig in figures:
        os.rename(fig, fig.replace('.svg', '_old.svg'))

for nb in tqdm(notebooks):
    print(f"Running {nb}")
    subprocess.run([
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--stdout",
        nb
    ], check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT)