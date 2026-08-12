# Whatif-Andes

## Description
Whatif-Andes is a reproducible workflow to assess climate-change what-if impacts on Andean water resources, combining glacier modelling (OGGM), hydrological modelling (LSTM), drought diagnostics, and water stress analyses.

## Experimental design

The workflow contrasts a reference period (2000–2024) against future projections
(2025–2099) under three global warming levels, each backed by an ensemble of climate
models.

The domain spans the Andes from 5 °S to 56 °S (66–79 °W; Ecuador and northern Peru are
excluded) and results are summarised over six latitudinal subregions: Outer tropics
(8–20 °S), Desert Andes (20–30 °S), Central Andes (30–35 °S), North Patagonia
(35–45 °S), South Patagonia (45–52 °S) and Fuegian Andes (52–56 °S).

Streamflow droughts follow a variable-threshold definition: days below the 10th
percentile of the reference flow, with events separated by less than 14 days pooled
together and pooled events shorter than 7 days discarded.

All of these settings live in [`config.yml`](config.yml) and can be changed there.

## Project Structure

```
├── config.yml                # Scenarios, subregions, periods, drought parameters
├── data/                     # Input datasets (admin, climate, glacier, hydro)
├── preprocessing/            # Basin selection, attributes, climate scenarios, water demand
├── modelling/                # OGGM glacier runs and LSTM training/inference
├── postprocessing/           # Glacier outputs, drought and water stress analyses
├── figures/                  # Notebooks producing the manuscript figures and tables
├── src/                      # Shared library code
│   ├── droughts.py           # Threshold-based drought identification and extent
│   ├── glaciers.py           # OGGM climate inputs and basin-level aggregation
│   ├── lstm_inputs.py        # Builds the neuralhydrology dataset (attributes + forcings)
│   ├── plotting.py           # Shared Plotly style for the figures
│   └── utils.py              # Config loading, zonal statistics, model output handling
├── run_notebooks.py          # Interactive runner for the full workflow
├── pixi.toml                 # Environment definition
├── pyproject.toml            # Package metadata and linting configuration
└── LICENSE
```

## Installation

```bash
git clone https://github.com/rodaguayo/Whatif-Andes.git
cd Whatif-Andes
pixi install && pixi shell
```

For OGGM glacier modelling, a separate conda environment is required. See [massbalance-sandbox/README.md](massbalance-sandbox/README.md#how-to-install) for setup instructions, then run `pip install -e .` from the project root to make `src` importable.

CUDA GPU recommended for LSTM training.

## Data

AndeanGC, ERA5, DEM, glacier outlines, and HydroATLAS datasets must be downloaded
separately. Update [`config.yml`](config.yml) with their paths.

## Usage

The whole workflow is driven by a single interactive script, which executes the
notebooks in dependency order:

```bash
python run_notebooks.py
```

Stages are cumulative: preprocessing feeds the glacier model, whose outputs are forcings
for the LSTM, so run them in order the first time. Notebooks can also be run individually in Jupyter; `run_notebooks.py` only wraps `jupyter nbconvert --execute` and encodes the correct order.

## Citation

If you use this workflow, please cite the associated publication:

> ```bibtex
> @article{aguayo_glacier,
>   title={Glacier loss amplifies streamflow droughts and water stress in the Andes},
>   author={Aguayo, Rodrigo; Zekollari, Harry; Bolibar, Jordi; van Tiel, Marit; Ayala, Álvaro; Van Tricht, Lander and Ultee, Lizz},
>   journal={[TBC]},
>   year={[TBC]}
> }
> ```

## License

The code in this repository is released under the [GNU General Public License v3.0](LICENSE). Derivative works that redistribute this code must remain under the same license. The figures, tables and derived datasets produced by the workflow are released under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) (CC-BY-4.0): they may be reused and adapted provided the publication listed above is cited.

## Contact

Rodrigo Aguayo — Vrije Universiteit Brussel (VUB)  

## Acknowledgments

This project was developed with support of a junior postdoctoral fellowship (No. 1207526N) of the Research Foundation—Flanders (FWO). 