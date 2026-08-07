"""
Build the neuralhydrology (LSTM) input dataset.

Converts the raw project data into the folder layout expected by
neuralhydrology under ``<pipeline>/lstm/data``:

- ``attributes/basin_attributes.csv``: static attributes indexed by ``gauge_id``
- ``time_series/<basin_id>.nc``: one netCDF per basin with the observed
  discharge (``q_mm``), the ERA5 forcings and the OGGM glacier melt series
  (historical + one series per what-if scenario).

The entry point is :func:`prepare_lstm_inputs`, which chains the individual
steps and is meant to be called before the model runs.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Default forcing variables and file naming templates
CLIMATE_VARS = ["tasmax", "tasmin", "tas", "ep", "prcp"]
ANDEANGC_CLIMATE = "climate/historical/AndeanGC_{var}_ERA5_1960_2024.parquet"
HYDROATLAS_CLIMATE = "data/climate/HydroAtlas_{var}_ERA5_1990_2024.parquet"
HYDROATLAS_BASINS = "data/hydro/basins_hydroATLAS_lev08.gpkg"
GLACIER_DIR = "data/glacier"
GLACIER_PATTERN = "*historical*.nc"

# m3/s -> mm/d, given a basin area in km2
_M3S_TO_MMD = 1000 * 86400
_KM2_TO_M2 = 1e6


def get_output_dir(cfg, create=True):
    """
    Return (and optionally create) the neuralhydrology data directory.

    Parameters
    ----------
    cfg : dict
        Project configuration as returned by ``src.utils.load_config``.
    create : bool, optional
        Whether to create the ``attributes`` and ``time_series`` subfolders,
        by default True.

    Returns
    -------
    Path
        Path to ``<pipeline>/lstm/data``.
    """
    output_dir = Path(cfg["dir"]["pipeline"]) / "lstm" / "data"
    if create:
        (output_dir / "time_series").mkdir(parents=True, exist_ok=True)
        (output_dir / "attributes").mkdir(parents=True, exist_ok=True)
    return output_dir


def build_attributes(cfg, output_dir=None):
    """
    Combine the AndeanGC and HydroATLAS static attributes.

    Glacier volume is normalised by basin area and columns with missing values
    are dropped, since neuralhydrology cannot handle NaNs in the attributes.

    Parameters
    ----------
    cfg : dict
        Project configuration.
    output_dir : Path, optional
        Data directory to write ``attributes/basin_attributes.csv`` into.  If
        None, the file is not written.

    Returns
    -------
    pd.DataFrame
        Attributes indexed by ``gauge_id`` (AndeanGC basins first, then the
        HydroATLAS target basins).
    """
    metadata_ggc = pd.read_csv(Path(cfg["dir"]["andeangc"]) / "dataset/AndeanGC_metadata.csv", index_col=0)
    metadata_all = gpd.read_file(HYDROATLAS_BASINS).drop(columns=["geometry"]).set_index("gauge_id")
    metadata = pd.concat([metadata_ggc, metadata_all])

    # Normalization
    metadata["glacier_volume_M22"] = metadata.glacier_volume_M22 / metadata.basin_area

    metadata = metadata.dropna(axis=1)

    if output_dir is not None:
        metadata.to_csv(Path(output_dir) / "attributes" / "basin_attributes.csv")

    return metadata


def build_streamflow(cfg, metadata):
    """
    Load the observed AndeanGC discharge and convert it to mm/d.

    Parameters
    ----------
    cfg : dict
        Project configuration (uses ``time.ref_start`` / ``time.ref_end``).
    metadata : pd.DataFrame
        Attributes table with a ``basin_area`` column (km2) indexed by gauge id.

    Returns
    -------
    xr.DataArray
        Discharge named ``q_mm`` with dims ``("date", "basin_id")``.
    """
    data = pd.read_csv(Path(cfg["dir"]["andeangc"]) / "dataset/AndeanGC_data_1950_2024_qc.csv",
                       index_col=0, parse_dates=True)
    data = data.loc[cfg["time"]["ref_start"]:cfg["time"]["ref_end"]]
    data = (data * _M3S_TO_MMD) / (metadata.basin_area * _KM2_TO_M2)

    return xr.DataArray(
        data.values,
        coords={"date": data.index, "basin_id": data.columns},
        dims=("date", "basin_id"),
        name="q_mm")


def add_climate_forcing(data, cfg, metadata, variables=None):
    """
    Merge the ERA5 forcings (AndeanGC + HydroATLAS basins) into *data*.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Dataset to merge the forcings into (dims ``date`` / ``basin_id``).
    cfg : dict
        Project configuration (uses ``time.ref_start`` / ``time.ref_end``).
    metadata : pd.DataFrame
        Attributes table; its index defines the basin order.
    variables : list of str, optional
        Forcing variables to load, by default :data:`CLIMATE_VARS`.

    Returns
    -------
    xr.Dataset
        *data* with one variable added per forcing.
    """
    variables = CLIMATE_VARS if variables is None else variables

    for var in tqdm(variables):
        climate_gcc = pd.read_parquet(Path(cfg["dir"]["andeangc"]) / ANDEANGC_CLIMATE.format(var=var))
        climate_all = pd.read_parquet(HYDROATLAS_CLIMATE.format(var=var))
        climate = pd.concat([climate_gcc, climate_all], axis=1)

        climate = climate.loc[cfg["time"]["ref_start"]:cfg["time"]["ref_end"]]
        climate.index.name = "date"
        climate = climate[metadata.index]

        climate = xr.DataArray(
            climate.values,
            coords={"date": climate.index, "basin_id": climate.columns},
            dims=("date", "basin_id"),
            name=var)
        data = xr.merge([data, climate], join="outer")

    return data


def add_glacier_melt(data, metadata, glacier_dir=GLACIER_DIR, pattern=GLACIER_PATTERN):
    """
    Merge the OGGM glacier melt series (historical + what-if) into *data*.

    Each ``run_output_*.nc`` file becomes a variable named ``gmelt_*``.  Basins
    without glaciers are filled with zeros so every basin file has the same
    variables.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Dataset to merge the melt series into.
    metadata : pd.DataFrame
        Attributes table with a ``basin_area`` column (km2).
    glacier_dir : str or Path, optional
        Folder holding the OGGM outputs, by default ``"data/glacier"``.
    pattern : str, optional
        Glob pattern selecting the OGGM outputs, by default ``"*historical*.nc"``.

    Returns
    -------
    xr.Dataset
        *data* with one ``gmelt_*`` variable per matched file.
    """
    area = xr.DataArray(metadata.basin_area.values, coords={"gauge_id": metadata.index}, dims="gauge_id")

    for path in tqdm(sorted(Path(glacier_dir).glob(pattern))):
        name = path.name.replace("run_output_", "gmelt_").replace(".nc", "")

        q_glacier = xr.open_dataset(path).glacier_melt
        q_glacier = q_glacier.reindex(gauge_id=metadata.index, fill_value=0)
        q_glacier = (q_glacier * _M3S_TO_MMD) / (area * _KM2_TO_M2)
        q_glacier = q_glacier.rename({"gauge_id": "basin_id"}).rename(name)

        data = xr.merge([data, q_glacier], join="outer")

    return data


def write_basin_files(data, output_dir):
    """
    Write one compressed netCDF per basin in ``time_series``.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with a ``basin_id`` dimension.
    output_dir : Path
        Data directory containing the ``time_series`` subfolder.

    Returns
    -------
    list of Path
        Paths of the written files.
    """
    output_dir = Path(output_dir)
    paths = []

    for basin in tqdm(data.basin_id.values):
        data_basin = data.sel(basin_id=basin).astype("float32")
        path = output_dir / "time_series" / "{}.nc".format(basin)
        data_basin.to_netcdf(path, encoding={var: {"compression": "zstd"} for var in data_basin.data_vars})
        paths.append(path)

    return paths


def prepare_lstm_inputs(cfg, variables=None, write=True):
    """
    Build the complete neuralhydrology input dataset.

    Chains :func:`build_attributes`, :func:`build_streamflow`,
    :func:`add_climate_forcing`, :func:`add_glacier_melt` and
    :func:`write_basin_files`.  Rerunning it also restores the per-basin files
    after the what-if runs, which overwrite ``gmelt_historical`` in place.

    Parameters
    ----------
    cfg : dict
        Project configuration as returned by ``src.utils.load_config``.
    variables : list of str, optional
        Forcing variables to load, by default :data:`CLIMATE_VARS`.
    write : bool, optional
        Whether to write the attributes CSV and the per-basin netCDFs,
        by default True.

    Returns
    -------
    xr.Dataset
        The merged dataset with dims ``("date", "basin_id")``.
    """
    output_dir = get_output_dir(cfg) if write else None

    metadata = build_attributes(cfg, output_dir=output_dir)
    data = build_streamflow(cfg, metadata)
    data = add_climate_forcing(data, cfg, metadata, variables=variables)
    data = add_glacier_melt(data, metadata)

    if write:
        write_basin_files(data, output_dir)

    return data
