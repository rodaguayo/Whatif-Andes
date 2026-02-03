"""Module for identifying drought events in time series data."""
import pandas as pd

def find_droughts(xarray_data, threshold_quantile=0.1, suffix="", threshold_df=None):
    """
    Calculate low-flow metrics per gauge, including frequency of drought events
    and drought intensity (deficit / duration).

    Parameters:
    - xarray_data: xarray DataArray with dimensions ("date", "gauge_id")
    - threshold_quantile: quantile used for threshold (e.g., 0.1 for Q90)
    - suffix: suffix string appended to metric names
    - threshold_df: optional xarray DataArray used to calculate thresholds
                   (if None, uses xarray_data)

    Returns:
    - DataFrame with drought metrics per gauge.
    """
    
    if threshold_df is None:
        threshold_df = xarray_data
    
    results = []
    
    for gauge_id in xarray_data.gauge_id.values:
        flow_series = xarray_data.sel(gauge_id=gauge_id).to_series()
        
        # 1) Calculate threshold
        threshold_val = threshold_df.sel(gauge_id=gauge_id).to_series()
        threshold_val = threshold_val.quantile(threshold_quantile)
        Q90 = flow_series.quantile(threshold_quantile)

        # Metrics
        # 2-3) Min values 
        min_30day = flow_series.rolling(window=30, min_periods=1, center=True).mean().resample("YE").min().mean()
        min_value = flow_series.resample("YE").min().mean()

        # 4) Day-of-year of annual minimum flow (averaged across years)
        annual_min_doy = flow_series.resample("YE").apply(
            lambda s: s.idxmin().timetuple().tm_yday if s.notna().any() and not s.empty else pd.NA
        )
        doy_min = annual_min_doy.mean()


        ## 4) Count drought events
        is_drought = (flow_series <= threshold_val)

        events_per_year = is_drought & (~is_drought.shift(1, fill_value=False))
        events_per_year = events_per_year.resample("YE").sum().mean()

        # 5-7) Drought severity and intensity
        drought_values = flow_series.where(is_drought)
        severity = float((threshold_val - drought_values).sum(skipna=True))
        duration_total = is_drought.sum()
        intensity = float(severity / duration_total)

        results.append({
            'gauge_id': str(gauge_id),
            f'Q90{suffix}': float(Q90),  
            f'annual_min_value{suffix}': float(min_value),
            f'min_30day{suffix}': float(min_30day),
            f'doy_min{suffix}': int(doy_min),
            f'events_per_year{suffix}': float(events_per_year),
            f'severity{suffix}': severity,
            f'total_duration{suffix}': duration_total,
            f'intensity{suffix}': intensity,
        })
    
    df = pd.DataFrame(results).set_index('gauge_id')
    return df

def drought_area_timeseries(xarray_data, metadata, threshold_quantile=0.1, threshold_df=None):
    """
    Returns a time series of total area affected by droughts.
    
    Parameters:
    - xarray_data: xarray DataArray with dimensions ("date", "gauge_id")
    - metadata: DataFrame with basin_area indexed by gauge_id
    - threshold_quantile: quantile used for threshold (e.g., 0.1 for Q90)
    - threshold_df: optional xarray DataArray used to calculate thresholds
                   (if None, uses xarray_data)
    
    Returns:
    - DataFrame with dates as index and latitude as columns, containing drought areas
    """
    if threshold_df is None:
        threshold_df = xarray_data
    
    # Vectorized threshold calculation
    thresholds = threshold_df.quantile(threshold_quantile, dim='date')
    
    # Create drought mask
    is_drought = xarray_data <= thresholds
    
    # Convert to DataFrame for easier manipulation
    drought_df = is_drought.to_dataframe(name='is_drought').reset_index()
    
    # Add metadata
    drought_df = drought_df.merge(
        metadata[['basin_area', 'basin_cenlat']], 
        left_on='gauge_id', 
        right_index=True
    )
    
    # Filter only drought days and calculate area
    drought_df = drought_df[drought_df['is_drought']]
    drought_df['lat'] = drought_df['basin_cenlat'].astype(int)
    
    # Pivot to time series format
    ts = drought_df.pivot_table(
        index='date', 
        columns='lat', 
        values='basin_area', 
        aggfunc='sum',
        fill_value=0
    ).astype(int)
    
    return ts