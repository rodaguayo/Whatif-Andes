import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import xarray as xr
import xarray_regrid

variables = ["melt_off_glacier_daily", "melt_on_glacier_daily", "liq_prcp_off_glacier_daily", "liq_prcp_on_glacier_daily"]

def create_OGGM_climate(prcp, temp, hgt, output_path, y0=None, y1=None):
    """
    Create a NetCDF file from precipitation, temperature, and elevation datasets.

    Parameters:
    prcp (str): Path to precipitation dataset.
    temp (str): Path to temperature dataset.
    hgt (str): Path to elevation dataset.
    output_path (str): Path to save the NetCDF file.
    y0 (int, optional): Start year for subsetting the time period.
    y1 (int, optional): End year for subsetting the time period.
    """
    
    # Load datasets from paths
    prcp = xr.open_dataset(prcp)
    temp = xr.open_dataset(temp)
    hgt = xr.open_dataset(hgt)

    # Subset time period if y0 and y1 are provided
    if y0 is not None and y1 is not None:
        prcp = prcp.sel(time=slice(str(y0), str(y1)))
        temp = temp.sel(time=slice(str(y0), str(y1)))

    hgt = hgt.rename({'y':'lat', 'x':'lon', 'band_data':'hgt'})
    hgt = hgt.sel(band=1, drop=True).drop_vars("spatial_ref")

    # Ensure spatial alignment
    hgt = hgt.regrid.linear(prcp)
    temp = temp.regrid.linear(prcp)
    
    # Merge datasets
    ds = xr.merge([prcp, temp, hgt]).rename({'tas': 'temp'})
    ds = ds.rename({'lat':'latitude', 'lon':'longitude'})
    ds["hgt"] = ds["hgt"].fillna(0)

    # Assign units
    ds.prcp.attrs['units'] = "mm"
    ds.temp.attrs['units'] = "degC"
    ds.hgt.attrs['units'] = "m"
    ds.hgt.attrs['long_name'] = "elevation"
    
    # Save to NetCDF
    ds.to_netcdf(output_path)

def hydro_variables(ds: xr.Dataset, only_melt: bool = False) -> xr.Dataset:
    """
    Calculate hydrological variables from glacier model output.
    This function computes glacier melt and optionally glacier runoff from daily
    glacier model data, converting from mm/day to m³/s.
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing glacier model variables:
        - melt_on_glacier_daily: Daily melt on glacier surface (mm/day)
        - melt_off_glacier_daily: Daily melt off glacier surface (mm/day)
        - liq_prcp_on_glacier_daily: Daily liquid precipitation on glacier (mm/day)
        - liq_prcp_off_glacier_daily: Daily liquid precipitation off glacier (mm/day)
    only_melt : bool, optional
        If True, only compute and return glacier_melt variable.
        If False, compute both glacier_melt and glacier_runoff.
        Default is False.
    Returns
    -------
    xr.Dataset
        Dataset containing one or both of the following variables:
        - glacier_melt: Glacier melt rate (m³/s), clipped to non-negative values
        - glacier_runoff: Total glacier runoff including melt and liquid precipitation
          from both on and off glacier areas (m³/s), clipped to non-negative values
    Notes
    -----
    Conversion factor: 1e-3 / 86400 converts mm/day to m³/s
    (mm to m: 1e-3, day to seconds: 1/86400)
    """
    ds["glacier_melt"] = ds.melt_on_glacier_daily * 1e-3 / 86400  # m3/s
    ds["glacier_melt"] = ds.glacier_melt.clip(0)  # m3/s

    if only_melt:
        return ds[["glacier_melt"]]
    
    else:
        ds["glacier_runoff"] = (
            (ds.melt_off_glacier_daily + ds.melt_on_glacier_daily +
                ds.liq_prcp_off_glacier_daily + ds.liq_prcp_on_glacier_daily) * 1e-3) / 86400  # m3/s
        ds["glacier_runoff"] = ds.glacier_runoff.clip(0)  # m3/s
        return ds[["glacier_melt", "glacier_runoff"]]
    

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
        
        # Select data for these glaciers
        basin_data = oggm_ts.sel(rgi_id=rgi_ids)
        basin_data["glacier_melt"] = basin_data["glacier_melt"] * xr.DataArray(fractions, dims=['rgi_id'])
        
        # Sum and expand dimensions
        basin_sum = basin_data.sum(dim='rgi_id')
        basin_sum = basin_sum.expand_dims(gauge_id=[basin])
        basin_list.append(basin_sum)
    
    # Combine all basins
    result = xr.concat(basin_list, dim='gauge_id')
    result = result.rename({'datetime': 'date'})
    return result