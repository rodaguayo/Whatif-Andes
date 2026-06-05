# Whatif-Andes

## Description
Whatif-Andes is a reproducible workflow to assess climate-change what-if impacts on Andean water resources, combining glacier modelling (OGGM), hydrological modelling (LSTM), drought diagnostics, and water stress analyses.

## Project Structure

```
├── config.yml
├── data/
├── preprocessing/
│   ├── climate_scenarios.ipynb
│   ├── glacier_modelling_inputs_v15.ipynb
│   ├── glacier_modelling_inputs_v16.ipynb
│   ├── hydro_modelling_inputs.ipynb
│   ├── target_basins_attributes.ipynb
│   ├── target_basins_selection.ipynb
│   ├── training_validation_splitting.ipynb
│   └── water_demand.ipynb
├── modelling/
│   ├── hydrological_modelling_LSTM.yml
│   ├── hydro_modelling_LSTM_finetuning.ipynb
│   ├── hydro_modelling_LSTM_runs.ipynb
│   ├── glacier_modelling_OGGM_v15.ipynb
│   └── glacier_modelling_OGGM_v16.ipynb
├── postprocessing/
│   ├── drought_analysis.ipynb
│   ├── glacier_modelling_outputs.ipynb
│   └── water_stress_analysis.ipynb
├── figures/
│   ├── fig_*.ipynb
│   ├── table_summary.ipynb
│   └── generate_all_figures.py
└── src/
    ├── droughts.py
    ├── glaciers.py
    └── utils.py
```

## Requirements

- **Pixi** ([install guide](https://pixi.sh/latest/))
- Hardware with CUDA GPU recommended for LSTM training

All dependencies are listed in [`pixi.toml`](pixi.toml).
Key packages: PyTorch, NeuralHydrology, OGGM, Xarray, GeoPandas.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/rodaguayo/Whatif-Andes.git
cd Whatif-Andes

# 2. Create the environment,  install the project, and activate the env
pixi install
pixi shell
```

The AndeanGC, ERA5, DEM, glacier outlines, and HydroATLAS datasets must be downloaded
separately (see the `data/` directory structure in the notebooks). Update the paths
in [`config.yml`](config.yml) to point to your dataset locations.

## Usage

Workflow notebooks are organized into four stages and should be run in order:

```bash
# Stage 1 — Preprocessing
preprocessing/climate_scenarios.ipynb
preprocessing/target_basins_selection.ipynb
preprocessing/target_basins_attributes.ipynb
preprocessing/glacier_modelling_inputs_v15.ipynb
preprocessing/glacier_modelling_inputs_v16.ipynb
preprocessing/hydro_modelling_inputs.ipynb
preprocessing/training_validation_splitting.ipynb
preprocessing/water_demand_correction.ipynb
preprocessing/water_demand_formatting.ipynb

# Stage 2 — Modelling
modelling/glacier_modelling_OGGM_v15.ipynb
modelling/glacier_modelling_OGGM_v16.ipynb
modelling/hydro_modelling_LSTM_finetuning.ipynb
modelling/hydro_modelling_LSTM_runs.ipynb

# Stage 3 — Postprocessing
postprocessing/glacier_modelling_outputs.ipynb
postprocessing/drought_analysis.ipynb
postprocessing/water_stress_analysis.ipynb

# Stage 4 — Figures (batch)
python figures/generate_all_figures.py
```


## Citation

If you use this workflow, please cite the associated publication:
[TBD].  

## Contact

Rodrigo Aguayo — Vrije Universiteit Brussel (VUB)  

## Acknowledgments

This project was developed with support from [funding sources — TBD].  
```