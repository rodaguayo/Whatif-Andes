"""
Utility functions for data processing and spatial analysis.

This module provides functions for:
- Converting pickle files to different output formats
- Aggregating data by main basin
- Extracting attributes from rasters
- Extracting time series from rasters
"""

import numpy as np
import pandas as pd
import xarray as xr
import pickle
from tqdm import tqdm
from glob import glob
import yaml
import os
from pathlib import Path
import xagg as xa # faster for large netcdf files

def process_outputs(path_run, period, best_epoch, metrics=None):

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
    if isinstance(data, pd.DataFrame): # DataFrame case 
        
        data = data * shape.basin_area * 1e6 / (1000 * 86400)
        data.columns = shape.main_bas
        data = data.T.groupby(data.columns).sum().T
        data = (data * 1000 * 86400) / (shape.basin_area.groupby(shape.main_bas).sum() * 1e6)
        return data
    
    else:  # xarray DataArray case
       
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
    
    from exactextract import exact_extract # faster for large rasters 

    if isinstance(raster, str):
        raster = xr.open_dataset(raster)

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

xa.set_options(impl='for_loop')
def extract_timeseries(raster, shapefile, time_period=None):
    
    raster = raster.sel(lon=slice(shapefile.total_bounds[0], shapefile.total_bounds[2]),
                        lat=slice(shapefile.total_bounds[1], shapefile.total_bounds[3]))
    
    # Handle both 'time' and 'date' dimension names
    time_dim = 'time' if 'time' in raster.dims else 'date'
    if time_period is not None:
        raster = raster.sel({time_dim: slice(pd.to_datetime(time_period[0]), pd.to_datetime(time_period[1]))})
    
    weightmap = xa.pixel_overlaps(raster, shapefile, impl='for_loop', silent=True)
    ds = xa.aggregate(raster, weightmap, impl = "numba", silent = True)
    ds = ds.to_dataframe().astype("float32")[raster.name]
    ds = ds.round(3)
    ds = ds.unstack('poly_idx')
    ds.columns = shapefile.gauge_id
    ds.columns.name = None

    return ds


def compile_settings(working_dir):
    """
    Load YAML settings files and convert to DataFrame.
    
    Parameters
    ----------
    working_dir : str
        Root directory to search for settings_daily.yml files.
    
    Returns
    -------
    pd.DataFrame
        Settings data with rgi_id column.
    """
    
    file_paths = glob(working_dir + '/**/settings_daily.yml', recursive=True)
    records = []
    
    for fp in file_paths:
        with open(fp, 'r') as fh:
            data = yaml.safe_load(fh)
        
        # add file metadata
        data['rgi_id'] = os.path.basename(os.path.dirname(fp))
        records.append(data)
    
    df_settings = pd.json_normalize(records)
    return df_settings

def find_best_epoch(path_run, n_epoch, metric = "KGE"):
    """
    Find the best epoch based on validation metric.
    
    Returns:
        int: Best epoch number
    """

    metrics = []
    for epoch in range(1, n_epoch + 1):
        val_metric = pd.read_csv(path_run / f"validation/model_epoch{epoch:03d}/validation_metrics.csv")
        val_metric = val_metric[metric].median()
        metrics.append(val_metric)
    
    best_epoch = metrics.index(max(metrics)) + 1
    return best_epoch

def reshape_dataset(initial_dataset):
    """
    Reshape the dataset from (month, grid_num) to (month, lat, lon).

    Parameters:
        initial_dataset (xr.Dataset): Input dataset with 'variable', 'lat', 'lon' variables.

    Returns:
        xr.DataArray: Reshaped DataArray with dimensions (date, lat, lon).
    """
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
    Load commune CSV, convert from m3/s to mm/year, join to shape and convert to xarray via make_geocube.
    Returns (geo_df, xarray.DataArray).
    """
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