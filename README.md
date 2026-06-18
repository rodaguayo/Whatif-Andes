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

Notebooks should be run in order: preprocessing → modelling → postprocessing → figures.


## Citation

If you use this workflow, please cite the associated publication:

> ```bibtex
> @article{aguayo_glacier,
>   title={Glacier loss amplifies streamflow droughts and water stress in the Andes},
>   author={Aguayo, Rodrigo and Zekollari, Harry and Bolibar, Jordi and van Tiel, Marit and Ayala, Álvaro and Van Tricht, Lander and Ultee, Lizz},
>   journal={[TBD]},
>   year={2026}
> }
> ```

## Contact

Rodrigo Aguayo — Vrije Universiteit Brussel (VUB)  

## Acknowledgments

This project was developed with support from FWO.  
```