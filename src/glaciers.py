"""
Glacier data processing and analysis module.

This module provides functions for processing OGGM (Open Global Glacier Model) data,
including climate data preparation, hydrological variable calculations, and
basin-level aggregation.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import xarray as xr

variables = ["melt_off_glacier_daily", "melt_on_glacier_daily", "liq_prcp_off_glacier_daily",
             "liq_prcp_on_glacier_daily", "snowfall_on_glacier_daily"]

def create_OGGM_climate(prcp, temp, hgt, output_path, y0=None, y1=None, grid=None,
                        hgt_cell_mean=False):
    """
    Create a NetCDF file from precipitation, temperature, and elevation datasets.

    Parameters:
    prcp (str): Path to precipitation dataset.
    temp (str): Path to temperature dataset.
    hgt (str): Path to elevation dataset.
    output_path (str): Path to save the NetCDF file.
    y0 (int, optional): Start year for subsetting the time period.
    y1 (int, optional): End year for subsetting the time period.
    grid (str, optional): Path to a file with lat/lon coordinates (e.g. a GCM file). If given,
        all variables are linearly interpolated to that grid instead of the precipitation grid.
        The GDPCIR grid is shifted half a cell from ERA5, so each value becomes the mean of the
        four surrounding ERA5 cells, which is the ERA5 reference GDPCIR was bias-corrected
        against. Historical and GCM climate then share the same cells and reference height.
    hgt_cell_mean (bool, optional): If True, the reference height of each cell is the mean
        DEM elevation over the cell (sea = 0 m), i.e. the smooth terrain the gridded temperature
        represents. If False, the DEM value at the cell centre. The DEM pixels must nest in the
        cells.
    """
    
    # Load datasets from paths
    prcp = xr.open_dataset(prcp)
    temp = xr.open_dataset(temp)
    hgt = xr.open_dataset(hgt)

    # the ERA5 1960-2025 files name their coordinates latitude/longitude
    prcp = prcp.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in prcp.dims})
    temp = temp.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in temp.dims})

    # Subset time period if y0 and y1 are provided
    if y0 is not None and y1 is not None:
        prcp = prcp.sel(time=slice(str(y0), str(y1)))
        temp = temp.sel(time=slice(str(y0), str(y1)))

    hgt = hgt.rename({'y':'lat', 'x':'lon', 'band_data':'hgt'})
    hgt = hgt.sel(band=1, drop=True).drop_vars("spatial_ref")

    # Ensure spatial alignment (linear interpolation to the precipitation grid; plain xarray,
    # so this also runs in the OGGM env, which has no xarray-regrid)
    if grid is not None:
        with xr.open_dataset(grid) as g:
            lat, lon = g.lat.values, g.lon.values
        prcp = prcp.interp(lat=lat, lon=lon, method="linear")
    if hgt_cell_mean:
        dlat = abs(float(prcp.lat[1] - prcp.lat[0]))
        dlon = abs(float(prcp.lon[1] - prcp.lon[0]))
        dem = hgt.hgt.sortby("lat").sortby("lon")
        dem = dem.sel(lat=slice(float(prcp.lat.min()) - dlat / 2, float(prcp.lat.max()) + dlat / 2),
                      lon=slice(float(prcp.lon.min()) - dlon / 2, float(prcp.lon.max()) + dlon / 2))
        n = int(round(dlat / abs(float(dem.lat[1] - dem.lat[0]))))
        m = int(round(dlon / abs(float(dem.lon[1] - dem.lon[0]))))
        if dem.sizes["lat"] != n * prcp.sizes["lat"] or dem.sizes["lon"] != m * prcp.sizes["lon"]:
            raise ValueError("The DEM pixels do not nest in the climate grid cells")
        dem = dem.fillna(0).coarsen(lat=n, lon=m).mean()
        hgt = dem.interp(lat=prcp.lat, lon=prcp.lon, method="nearest").to_dataset(name="hgt")
    else:
        hgt = hgt.interp(lat=prcp.lat, lon=prcp.lon, method="linear")
    temp = temp.interp(lat=prcp.lat, lon=prcp.lon, method="linear")
    
    # Merge datasets
    ds = xr.merge([prcp, temp, hgt]).rename({'tas': 'temp'})
    ds = ds.rename({'lat':'latitude', 'lon':'longitude'})
    ds["hgt"] = ds["hgt"].fillna(0)
    ds["prcp"] = ds["prcp"].astype("float32")  # interp returns float64 (file twice as large)
    ds["temp"] = ds["temp"].astype("float32")

    # Assign units
    ds.prcp.attrs['units'] = "mm"
    ds.temp.attrs['units'] = "degC"
    ds.hgt.attrs['units'] = "m"
    ds.hgt.attrs['long_name'] = "elevation"
    
    # Save to NetCDF
    ds.to_netcdf(output_path)


def sync_gcm_ref_hgt(gdir, historical_file="climate_historical_daily_ERA5.nc",
                     gcm_pattern="gcm_data_daily_*.nc"):
    """
    Copy the calibrated reference height of the historical climate file to the GCM files.

    When no melt factor within its bounds matches the geodetic mass balance,
    MBsandbox's ``melt_f_calib_geod_prep_inversion`` shifts ``ref_hgt`` (a temperature
    correction) and writes it only into the historical climate file. The GCM files keep the
    uncorrected height of the grid cell, and ``run_from_climate_data_TIModel`` does not check
    ``gcm_data``, so without this step the future runs use a different temperature than the
    one the glacier was calibrated with (up to several °C).

    TIModel reads ``ref_hgt`` from the file attributes, and calibration restarts from
    ``uncorrected_ref_hgt``, so this is idempotent. Run it after calibration and before any
    run that reads ``gcm_data``. Meant for ``workflow.execute_entity_task``.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
        Glacier directory.
    historical_file : str, optional
        Name of the calibrated historical climate file in the glacier directory.
    gcm_pattern : str, optional
        Glob pattern of the GCM climate files in the glacier directory.

    Returns
    -------
    float
        The shift applied to the GCM files (corrected minus uncorrected height, m). NaN if the
        historical file is missing or a GCM file has a different grid-cell height (that file
        is then left untouched); it does not raise, so one glacier cannot stop a workflow.
    """
    from pathlib import Path
    import netCDF4

    hist_path = Path(gdir.dir) / historical_file
    if not hist_path.exists():
        return np.nan
    with netCDF4.Dataset(hist_path) as nc:
        ref_hgt = float(nc.ref_hgt)
        uncorrected = float(getattr(nc, "uncorrected_ref_hgt", nc.ref_hgt))

    shift = ref_hgt - uncorrected
    for path in sorted(Path(gdir.dir).glob(gcm_pattern)):
        with netCDF4.Dataset(path, "a") as nc:
            gcm_uncorrected = float(getattr(nc, "uncorrected_ref_hgt", nc.ref_hgt))
            if not np.isclose(gcm_uncorrected, uncorrected, atol=1):
                shift = np.nan
                continue
            nc.ref_hgt = ref_hgt
            nc.uncorrected_ref_hgt = uncorrected

    return shift



def hydro_variables(ds: xr.Dataset, only_melt: bool = False) -> xr.Dataset:
    """
    Calculate hydrological variables from glacier model output.
    This function computes glacier melt, on-glacier snowfall and optionally glacier
    runoff from daily glacier model data, converting from kg/day to m³/s.
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing glacier model variables (kg/day per glacier):
        - melt_on_glacier_daily: Daily melt on glacier surface (snow + ice)
        - melt_off_glacier_daily: Daily melt off glacier surface
        - liq_prcp_on_glacier_daily: Daily liquid precipitation on glacier
        - liq_prcp_off_glacier_daily: Daily liquid precipitation off glacier
        - snowfall_on_glacier_daily: Daily snowfall on glacier surface
    only_melt : bool, optional
        If True, return only the inputs of the hydrological model (glacier_melt and
        glacier_snow). If False, also compute glacier_runoff. Default is False.
    Returns
    -------
    xr.Dataset
        Dataset containing the following variables:
        - glacier_melt: Glacier melt rate (m³/s), clipped to non-negative values
        - glacier_snow: On-glacier snowfall (m³/s), clipped to non-negative values.
          Glacier melt includes the melt of this snow, so it is subtracted from the
          precipitation of the conceptual model to avoid counting it twice.
        - glacier_runoff (only if only_melt=False): Total glacier runoff including melt
          and liquid precipitation from both on and off glacier areas (m³/s), clipped
          to non-negative values
    Notes
    -----
    Conversion factor: 1e-3 / 86400 converts kg/day (of water) to m³/s
    (kg to m³: 1e-3, day to seconds: 1/86400)
    """
    ds["glacier_melt"] = ds.melt_on_glacier_daily * 1e-3 / 86400  # m3/s
    ds["glacier_melt"] = ds.glacier_melt.clip(0)  # m3/s
    ds["glacier_snow"] = ds.snowfall_on_glacier_daily * 1e-3 / 86400  # m3/s
    ds["glacier_snow"] = ds.glacier_snow.clip(0)  # m3/s

    if only_melt:
        return ds[["glacier_melt", "glacier_snow"]]

    else:
        ds["glacier_runoff"] = (
            (ds.melt_off_glacier_daily + ds.melt_on_glacier_daily +
                ds.liq_prcp_off_glacier_daily + ds.liq_prcp_on_glacier_daily) * 1e-3) / 86400  # m3/s
        ds["glacier_runoff"] = ds.glacier_runoff.clip(0)  # m3/s
        return ds[["glacier_melt", "glacier_snow", "glacier_runoff"]]
    

def julian_to_gregorian(year, julian_day):

    year = int(year)
    julian_day = int(julian_day)
    gregorian_date = pd.to_datetime(f'{int(year)}-01-01')
    gregorian_date += pd.to_timedelta(julian_day - 1, unit='D')
    gregorian_date = np.array(gregorian_date, dtype='datetime64[ns]')
    return gregorian_date

def change_calendar(ds: xr.Dataset) -> xr.Dataset:

    ds = ds.drop_vars(['hydro_year', 'hydro_month', 'calendar_year', 'calendar_month', 'calendar_day_2d'])
    ds = ds.stack(datetime=("time", "day_2d"))
    dates = [julian_to_gregorian(year, day) for year, day in ds.datetime.values]
    ds = ds.drop_vars(["datetime", 'time', 'day_2d'])
    ds["datetime"] = dates
    ds = ds.dropna("datetime", how = "all")
    return ds

def calculate_basin_overlap(glaciers_gdf, basins_gdf):
    """
    Calculate overlap between glaciers and basins.
    
    Parameters:
    -----------
    glaciers_gdf : GeoDataFrame
        GeoDataFrame containing glacier geometries
    basins_gdf : GeoDataFrame
        GeoDataFrame containing basin geometries with 'gauge_id' in index
        
    Returns:
    --------
    GeoDataFrame
        DataFrame with gauge_id and overlap_fraction for each glacier
    """
    # Reproject to EPSG:32719
    glaciers_gdf = glaciers_gdf.to_crs("EPSG:32719")
    basins_gdf = basins_gdf.to_crs("EPSG:32719")
    
    overlap = gpd.sjoin(glaciers_gdf, basins_gdf, how="left", predicate="intersects")
    overlap = overlap.dropna()
    
    overlap['glacier_area'] = overlap.geometry.area
    overlap['intersection_area'] = overlap.apply(
        lambda row: row.geometry.intersection(basins_gdf.loc[row.gauge_id, 'geometry']).area, axis=1)
    overlap['overlap_fraction'] = overlap['intersection_area'] / overlap['glacier_area']
    overlap = overlap[['gauge_id', 'overlap_fraction']]
    
    return overlap

def aggregate_data_by_basin(oggm_ts, basin_lookup):
    """
    Aggregate glacier data by basin, accounting for overlap fractions.
    
    Parameters:
    -----------
    oggm_ts : xr.Dataset
        OGGM time series data with rgi_id dimension
    basin_lookup : pd.DataFrame
        DataFrame with RGIId as index and columns 'gauge_id' and 'overlap_fraction' (ALREADY FILTERED)
        
    Returns:
    --------
    xr.Dataset
        Aggregated dataset with gauge_id dimension and date coordinate
    """
    
    # Get unique basins
    basins = basin_lookup['gauge_id'].unique()
    basin_list = []
    
    for basin in basins:
        # Get all glaciers in this basin with their overlap fractions
        basin_glaciers = basin_lookup[basin_lookup['gauge_id'] == basin]
        rgi_ids = basin_glaciers.index.tolist()
        fractions = basin_glaciers['overlap_fraction'].values
        
        # Select data for these glaciers and weight every variable by its overlap fraction
        basin_data = oggm_ts.sel(rgi_id=rgi_ids)
        basin_data = basin_data * xr.DataArray(fractions, dims=['rgi_id'])
        
        # Sum and expand dimensions
        basin_sum = basin_data.sum(dim='rgi_id')
        basin_sum = basin_sum.expand_dims(gauge_id=[basin])
        basin_list.append(basin_sum)
    
    # Combine all basins
    result = xr.concat(basin_list, dim='gauge_id')
    result = result.rename({'datetime': 'date'})
    return result