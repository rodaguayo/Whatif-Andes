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
[List software dependencies and requirements]

## Installation
[Provide installation instructions]

## Usage
[Explain how to use the project, run analyses, generate figures, etc.]

## Configuration
The project uses [config.yml](config.yml) for configuration settings.

## Figures
Figure notebooks are in `figures/`. Batch execution is handled by `figures/generate_all_figures.py` (`figures/*.ipynb`).

## Citation
[Add citation information if applicable]

## Contact
Rodrigo Aguayo (VUB)

## Acknowledgments
[Add acknowledgments if applicable]
```