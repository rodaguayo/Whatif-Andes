# Whatif-Andes

## Description
Whatif-Andes is a reproducible workflow to assess climate-change what-if impacts on Andean water resources, combining glacier modelling (OGGM), hydrological modelling (LSTM), drought diagnostics, and water stress analyses.

## Project Structure

```
├── config.yml
├── data/
├── preprocessing/
├── modelling/
├── postprocessing/
├── figures/
├── src/
├── run_notebooks.py
├── pixi.toml
├── pyproject.toml
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

## Citation

If you use this workflow, please cite the associated publication:

> ```bibtex
> @article{aguayo_glacier,
>   title={Glacier loss amplifies streamflow droughts and water stress in the Andes},
>   author={Aguayo, Rodrigo; Zekollari, Harry; Bolibar, Jordi; van Tiel, Marit; Ayala, Álvaro; Van Tricht, Lander and Ultee, Lizz},
>   journal={[TBD]},
>   year={2026}
> }
> ```

## Contact

Rodrigo Aguayo — Vrije Universiteit Brussel (VUB)  

## Acknowledgments

This project was developed with support from FWO.  
```