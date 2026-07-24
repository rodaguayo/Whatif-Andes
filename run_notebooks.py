import subprocess
import glob
import os
from tqdm import tqdm

from src.utils import load_config

config = load_config()

# Define notebook categories (excludes v16 and test variants)
CATEGORIES = {
    "Preprocessing": sorted([
        "preprocessing/climate_scenarios.ipynb",
        "preprocessing/target_basins_selection.ipynb",
        "preprocessing/target_basins_attributes.ipynb",
        "preprocessing/training_validation_splitting.ipynb",
        "preprocessing/water_demand_formatting.ipynb",
        "preprocessing/water_demand_correction.ipynb",
    ]),
    "Glacier modelling": sorted([
        "preprocessing/glacier_modelling_inputs.ipynb",
        "modelling/glacier_modelling_OGGM.ipynb",
        "postprocessing/glacier_modelling_outputs.ipynb",
    ]),
    "Hydrological modelling": sorted([
        "preprocessing/hydro_modelling_inputs.ipynb",
        "modelling/hydro_modelling_LSTM_runs.ipynb",
        "postprocessing/drought_analysis.ipynb",
        "postprocessing/water_stress_analysis.ipynb",
    ]),
    "Figures": sorted(glob.glob("figures/fig_*.ipynb") + glob.glob("figures/table_*.ipynb")),
}

CATEGORY_ORDER = ["Preprocessing", "Glacier modelling", "Hydrological modelling", "Figures"]


def run_notebook(nb):
    print(f"  Running {nb}")
    subprocess.run([
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--stdout",
        nb
    ], check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT)


def backup_figures():
    figures = sorted([
        f for f in glob.glob("figures/*.svg")
        if "slide" not in f and "workflow" not in f and "_old" not in f
    ])
    for fig in figures:
        os.rename(fig, fig.replace(".svg", "_old.svg"))
    print(f"  Backed up {len(figures)} figures")


def parse_choices(raw):
    parts = raw.replace(",", " ").split()
    choices = []
    for p in parts:
        if p.isdigit():
            choices.append(int(p))
    return choices


def main():
    print("\n=== Run Notebooks ===\n")
    print("Select notebooks to run:\n")
    print("  1. Preprocessing")
    print("  2. Glacier modelling")
    print("  3. Hydrological modelling")
    print("  4. Figures")
    print("  5. All")
    print("  0. Exit\n")

    raw = input("Enter choices (e.g., 1 3 or 1,3 or 5): ")
    choices = parse_choices(raw)

    if not choices or 0 in choices:
        print("Nothing to run.")
        return

    # Expand "All" to individual categories
    if 5 in choices:
        selected = list(CATEGORY_ORDER)
    else:
        selected = []
        for c in choices:
            if 1 <= c <= len(CATEGORY_ORDER):
                selected.append(CATEGORY_ORDER[c - 1])

    if not selected:
        print("Invalid selection.")
        return

    # Collect notebooks in order
    notebooks = []
    for cat in selected:
        notebooks.extend(CATEGORIES[cat])

    # Backup figures if Figures category is selected
    if "Figures" in selected:
        backup = input("\nRename existing figures to _old for comparison? (y/n): ").lower() == "y"
        if backup:
            backup_figures()

    # Run
    print(f"\nRunning {len(notebooks)} notebooks...\n")
    for nb in tqdm(notebooks):
        run_notebook(nb)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
