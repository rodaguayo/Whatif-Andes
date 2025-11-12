import xarray as xr
import xarray_regrid

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
    hgt = hgt.sel(band=1, drop=True).drop_vars("spatial_ref").fillna(0)

    # Ensure spatial alignment
    hgt = hgt.regrid.linear(prcp)
    temp = temp.regrid.linear(prcp)
    
    # Merge datasets
    ds = xr.merge([prcp, temp, hgt]).rename({'tas': 'temp'})
    ds = ds.rename({'lat':'latitude', 'lon':'longitude'})

    # Assign units
    ds.prcp.attrs['units'] = "mm"
    ds.temp.attrs['units'] = "degC"
    ds.hgt.attrs['units'] = "m"
    
    # Save to NetCDF
    ds.to_netcdf(output_path)
