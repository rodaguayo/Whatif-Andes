"""
Utility functions for data processing and spatial analysis.

This module provides functions for:
- Converting pickle files to different output formats
- Aggregating data by main basin
- Extracting attributes from rasters
- Extracting time series from rasters
"""

from glob import glob
import yaml
import os
from pathlib import Path

def load_config(config_path="config.yml", chdir=True):
    """Load project configuration, changing to project root if needed.

    Parameters
    ----------
    config_path : str, optional
        Path to config file relative to project root, by default "config.yml"
    chdir : bool, optional
        Whether to change to the project root directory, by default True

    Returns
    -------
    dict
        Parsed configuration
    """
    root = Path(__file__).resolve().parent.parent
    if chdir:
        os.chdir(root)
    with open(root / config_path) as f:
        config = yaml.safe_load(f)
    if "dir" in config:
        config["dir"] = {k: os.path.expanduser(v) for k, v in config["dir"].items()}
    return config

def process_outputs(path_run, period, best_epoch, metrics=None):
    """
    Run neuralhydrology evaluation and collect simulated discharge time series.

    Parameters
    ----------
    path_run : Path
        Path to a neuralhydrology run directory containing ``config.yml``.
    period : str
        Evaluation period passed to the tester (e.g. ``"test"`` or ``"validation"``).
    best_epoch : int
        Model epoch to load weights from.
    metrics : list of str, optional
        Metric names to compute and return alongside the time series.

    Returns
    -------
    xr.DataArray or tuple[xr.DataArray, pd.DataFrame]
        Simulated discharge (mm/d, clipped to 0) as a DataArray with a
        ``gauge_id`` dimension.  When *metrics* is provided, a DataFrame of
        per-basin metric values is returned as a second element.
    """

    import xarray as xr
    import pandas as pd
    from tqdm import tqdm
    from neuralhydrology.evaluation import get_tester
    from neuralhydrology.utils.config import Config

    time_series = []
    basin_ids = []
    metrics_dict = {metric: [] for metric in metrics} if metrics else {}
    
    tester = get_tester(cfg=Config(path_run / "config.yml"), run_dir=path_run, period=period, init_model=True)
    data = tester.evaluate(epoch=best_epoch, save_results=False, save_all_output=False, metrics=metrics)

    for basin_id, dataset_xr in tqdm(data.items(), leave=False):
        q_sim = dataset_xr['1D']['xr'].isel(time_step=0).drop_vars('time_step').q_mm_sim.clip(0)
        time_series.append(q_sim)
        basin_ids.append(basin_id)
        
        # Extract metrics if provided
        if metrics:
            for metric in metrics:
                metrics_dict[metric].append(dataset_xr['1D'][metric])
    
    ts_output = xr.concat(time_series, dim="gauge_id").assign_coords(gauge_id=basin_ids)
    metrics_output = pd.DataFrame(metrics_dict, index=basin_ids) if metrics else None
    return (ts_output, metrics_output) if metrics else ts_output
    
def aggregate_by_main_basin(data, shape):
    """
    Aggregate glacier melt or runoff time series by main basin.

    Parameters
    ----------
    data : pd.DataFrame or xr.DataArray
        Time series (columns: gauge_id, index: date) or DataArray with gauge_id coord.
    shape : gpd.GeoDataFrame
        HydroATLAS basin shapes with 'basin_area' and 'main_bas' columns.

    Returns
    -------
    pd.DataFrame or xr.DataArray
        Aggregated time series by main basin (columns/coord: main_bas, index/coord: date).
    """
    import pandas as pd
    if isinstance(data, pd.DataFrame): # DataFrame case 
        
        data = data * shape.basin_area * 1e6 / (1000 * 86400)
        data.columns = shape.main_bas
        data = data.T.groupby(data.columns).sum().T
        data = (data * 1000 * 86400) / (shape.basin_area.groupby(shape.main_bas).sum() * 1e6)
        return data
    
    else:  # xarray DataArray case
        import xarray as xr
        import pandas as pd
        
        # Convert to m3/s
        area = xr.DataArray(shape.basin_area.values, coords={"gauge_id": shape.index})
        data = data * area * 1e6 / (1000 * 86400)
        
        # Assign main_bas as new gauge_id
        data = data.assign_coords(gauge_id=shape.main_bas)
        data = data.groupby("gauge_id").sum()

        # Convert back to mm/d
        basin_area_sum = pd.Series(shape.basin_area.values, index=shape.main_bas).groupby(level=0).sum()
        basin_area_sum = xr.DataArray(basin_area_sum.values, coords={"main_bas": basin_area_sum.index})
        basin_area_sum = basin_area_sum.rename({"main_bas": "gauge_id"})
        data = (data * 1000 * 86400) / (basin_area_sum * 1e6)
        return data 
    

def extract_attributes(raster, shapefile, attribute_name, fun="mean"):
    """
    Extract zonal statistics from a raster for each basin polygon.

    Parameters
    ----------
    raster : xr.DataArray
        Raster dataset with spatial dimensions named ``lon``/``lat`` or ``x``/``y``.
    shapefile : gpd.GeoDataFrame
        Basin polygons with a ``gauge_id`` column used as the index.
    attribute_name : str
        Name assigned to the extracted column in the returned GeoDataFrame.
    fun : str, optional
        Aggregation function passed to ``exactextract`` (e.g. ``"mean"``, ``"sum"``),
        by default ``"mean"``.

    Returns
    -------
    gpd.GeoDataFrame
        *shapefile* with an additional column named *attribute_name*.
    """
    import xarray as xr
    from exactextract import exact_extract

    if "lon" in raster.dims and "lat" in raster.dims:
        raster = raster.sel(lon=slice(shapefile.total_bounds[0], shapefile.total_bounds[2]),
                            lat=slice(shapefile.total_bounds[1], shapefile.total_bounds[3]))
    elif "x" in raster.dims and "y" in raster.dims:
        raster = raster.sel(x=slice(shapefile.total_bounds[0], shapefile.total_bounds[2]),
                            y=slice(shapefile.total_bounds[3], shapefile.total_bounds[1]))

    if "lat" in raster.dims or "lon" in raster.dims:
        raster = raster.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

    ds = exact_extract(raster.load(), shapefile, fun, include_cols=["gauge_id"], 
                       output="pandas", progress=False)
    ds = ds.set_index("gauge_id")
    ds.rename(columns={fun: attribute_name}, inplace=True)
    shape = shapefile.join(ds[attribute_name], on="gauge_id")

    return shape

def extract_timeseries(raster, shapefile, time_period=None):
    """
    Extract area-weighted time series from a raster for each basin polygon.

    Parameters
    ----------
    raster : xr.DataArray
        Raster with ``lon``/``lat`` spatial dimensions and a ``time`` or ``date``
        dimension.
    shapefile : gpd.GeoDataFrame
        Basin polygons with a ``gauge_id`` column.
    time_period : tuple of str, optional
        ``(start, end)`` date strings used to subset the raster along the time
        dimension before aggregation.

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and ``gauge_id`` values as columns,
        rounded to 3 decimal places.
    """
    import pandas as pd
    import xarray as xr
    import xagg as xa
    
    raster = raster.sel(lon=slice(shapefile.total_bounds[0], shapefile.total_bounds[2]),
                        lat=slice(shapefile.total_bounds[1], shapefile.total_bounds[3]))
    
    # Handle both 'time' and 'date' dimension names
    time_dim = 'time' if 'time' in raster.dims else 'date'
    if time_period is not None:
        raster = raster.sel({time_dim: slice(pd.to_datetime(time_period[0]), pd.to_datetime(time_period[1]))})
    
    weightmap = xa.pixel_overlaps(raster, shapefile, impl='for_loop', silent=True)
    ds = xa.aggregate(raster, weightmap, impl="numba", silent=True)
    ds = ds.to_dataframe().astype("float32")[raster.name]
    ds = ds.round(3)
    ds = ds.unstack('poly_idx')
    ds.columns = shapefile.gauge_id
    ds.columns.name = None

    return ds


def compile_settings(working_dir):
    """
    Collect ``settings_daily.yml`` files from an OGGM working directory into a DataFrame.

    Parameters
    ----------
    working_dir : str
        Root directory to search recursively for ``settings_daily.yml`` files.

    Returns
    -------
    pd.DataFrame
        Flattened (json-normalized) settings for all glaciers found, with an
        additional ``rgi_id`` column derived from the parent directory name.
    """
    import pandas as pd

    file_paths = glob(working_dir + '/**/settings_daily.yml', recursive=True)
    records = []
    
    for fp in file_paths:
        with open(fp, 'r') as fh:
            data = yaml.safe_load(fh)
        
        data['rgi_id'] = os.path.basename(os.path.dirname(fp))
        records.append(data)
    
    df_settings = pd.json_normalize(records)
    return df_settings

def find_best_epoch(path_run, n_epoch, metric="KGE"):
    """
    Identify the training epoch with the highest median validation metric.

    Parameters
    ----------
    path_run : Path
        neuralhydrology run directory containing per-epoch validation CSV files
        at ``validation/model_epoch{NNN}/validation_metrics.csv``.
    n_epoch : int
        Total number of epochs to evaluate.
    metric : str, optional
        Column name of the metric to maximise, by default ``"KGE"``.

    Returns
    -------
    int
        1-based epoch number with the highest median *metric* value.
    """
    import pandas as pd

    metrics = []
    for epoch in range(1, n_epoch + 1):
        val_metric = pd.read_csv(path_run / f"validation/model_epoch{epoch:03d}/validation_metrics.csv")
        val_metric = val_metric[metric].median()
        metrics.append(val_metric)
    
    best_epoch = metrics.index(max(metrics)) + 1
    return best_epoch

def reshape_dataset(initial_dataset):
    """
    Reshape a flat (point-based) xarray Dataset into a gridded DataArray.

    Converts a Dataset with 1-D ``lat``/``lon`` coordinates (one row per grid
    cell) into a 3-D ``(date, lat, lon)`` DataArray.  The ``month`` dimension is
    renamed to ``date`` and aligned to a monthly ``DatetimeIndex`` starting
    1971-01-01.

    Parameters
    ----------
    initial_dataset : xr.Dataset
        Dataset with ``lat``, ``lon``, and ``month`` coordinates and a single
        data variable.

    Returns
    -------
    xr.DataArray
        Gridded DataArray with dimensions ``(date, lat, lon)``.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    lats = np.unique(initial_dataset['lat'].values)
    lons = np.unique(initial_dataset['lon'].values)
    nlat = len(lats)
    nlon = len(lons)
    var_name = list(initial_dataset.data_vars)[0]

    latlon_df = pd.DataFrame({'lat': initial_dataset['lat'].values, 'lon': initial_dataset['lon'].values})
    lat_idx = latlon_df['lat'].map({v: i for i, v in enumerate(lats)}).values
    lon_idx = latlon_df['lon'].map({v: i for i, v in enumerate(lons)}).values

    reshaped = np.full((initial_dataset.sizes['month'], nlat, nlon), np.nan)
    reshaped[:, lat_idx, lon_idx] = initial_dataset[var_name].values

    reshaped = xr.DataArray(
        reshaped,
        dims=('month', 'lat', 'lon'),
        coords={'month': initial_dataset['month'], 'lat': lats, 'lon': lons},
        name=var_name
    )

    # Convert the 'month' coordinate to datetime (01-MM-YYYY) format
    month_dates = pd.date_range(start=pd.Timestamp("1971-01-01"), periods=480, freq='MS')
    reshaped = reshaped.assign_coords(month=month_dates)
    reshaped = reshaped.rename({'month': 'date'})
    return reshaped

def load_and_convert_csv_to_geocube(file, var_name, years=None, shape=None, res=None):
    """
    Load water-demand CSV data, convert units, and rasterise to a geocube DataArray.

    The CSV is expected to have comunas as rows and years as columns.  Values
    are converted from m³/s to mm/year using each commune's area before being
    rasterised at the requested resolution.

    Parameters
    ----------
    file : str, Path, or pd.DataFrame
        Path to a CSV file (index = ``cod_comuna``, columns = years) or a
        pre-loaded DataFrame with the same structure.
    var_name : str
        Name assigned to the output DataArray.
    years : list, optional
        Column names (year labels) to rasterise.
    shape : gpd.GeoDataFrame, optional
        Commune polygons with an ``area_m2`` column, joined to the DataFrame
        before rasterisation.
    res : int or float, optional
        Output raster resolution passed to ``make_geocube``.

    Returns
    -------
    xr.DataArray
        Rasterised DataArray with a ``time`` dimension corresponding to *years*.
    """
    import pandas as pd
    import xarray as xr
    from geocube.api.core import make_geocube

    if isinstance(file, (str, Path)):
        df = pd.read_csv(file, index_col=0)
    else:
        df = file
    df.index.name = "cod_comuna"
    df.index = range(1960, 2021)
    df = df.transpose()
    df = shape.join(df)

    # vectorized conversion: m3/s -> mm/year
    factor = 86400 * 365 * 1000  # seconds per day * days per year * mm per m
    df[years] = df[years].div(df["area_m2"], axis=0) * factor

    xr_ds = make_geocube(vector_data=df, measurements=years, resolution=res)
    xr_da = xr_ds.to_array(dim='time', name=var_name)
    xr_da = xr_da.drop_vars('spatial_ref', errors='ignore')
    return xr_da

def weighted_mean_by_lat(gdf, weight_col="basin_area", lat_col="basin_cenlat_rounded"):
    """
    Compute area-weighted means of all numeric columns grouped by latitude band.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame or pd.DataFrame
        Input data containing numeric columns, a weight column, and a latitude
        grouping column.
    weight_col : str, optional
        Column used as weights, by default ``"basin_area"``.
    lat_col : str, optional
        Column used to form latitude groups, by default
        ``"basin_cenlat_rounded"``.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by *lat_col* with weighted-mean values for every
        numeric column (excluding *weight_col* and *lat_col* themselves).
    """
    import pandas as pd

    numeric_cols = gdf.select_dtypes(include='number').columns.tolist()
    cols = [col for col in numeric_cols if col not in [weight_col, lat_col]]

    def _weighted_mean(group):
        weights = group[weight_col]
        return pd.Series({col: (group[col] * weights).sum() / weights.sum() for col in cols})
    
    return gdf.groupby(lat_col).apply(_weighted_mean, include_groups=False)