"""Module for identifying drought events in time series data."""
import pandas as pd

def _calculate_gauge_drought_mask_and_threshold(flow_series, threshold_series, threshold_quantile=0.1, pooling_window=None, min_duration=None, seasonal_threshold=False):
    """
    Calculate the threshold values and boolean drought mask for a single gauge time series.
    """
    if seasonal_threshold:
        threshold_ma = threshold_series.rolling(window=30, min_periods=1, center=True).mean()
        daily_q = threshold_ma.groupby(threshold_ma.index.dayofyear).quantile(threshold_quantile)
        threshold_val = pd.Series(
            flow_series.index.dayofyear.map(daily_q).values,
            index=flow_series.index
        )
    else:
        threshold_val = threshold_series.quantile(threshold_quantile)

    is_drought = (flow_series <= threshold_val)

    # Inter-event time pooling: merge drought events separated by short gaps
    if pooling_window is not None:
        drought_array = is_drought.values.copy()
        n = len(drought_array)
        i = 0
        while i < n:
            if not drought_array[i]:
                j = i
                while j < n and not drought_array[j]:
                    j += 1
                # Only fill if gap is between two drought periods (not at edges)
                if (j - i) <= pooling_window and i > 0 and j < n:
                    drought_array[i:j] = True
                i = j
            else:
                i += 1
        is_drought = pd.Series(drought_array, index=is_drought.index)

    # Minimum duration filtering: remove events shorter than min_duration days
    if min_duration is not None:
        drought_array = is_drought.values.copy()
        n = len(drought_array)
        i = 0
        while i < n:
            if drought_array[i]:
                j = i
                while j < n and drought_array[j]:
                    j += 1
                if (j - i) < min_duration:
                    drought_array[i:j] = False
                i = j
            else:
                i += 1
        is_drought = pd.Series(drought_array, index=is_drought.index)

    return is_drought, threshold_val

def find_droughts(xarray_data, threshold_quantile=0.1, suffix="", threshold_df=None, pooling_window=None, min_duration=None, seasonal_threshold=False):
    """
    Calculate low-flow metrics per gauge, including frequency of drought events
    and drought intensity (deficit / duration).

    Parameters:
    - xarray_data: xarray DataArray with dimensions ("date", "gauge_id")
    - threshold_quantile: quantile used for threshold (e.g., 0.1 for Q90)
    - suffix: suffix string appended to metric names
    - threshold_df: optional xarray DataArray used to calculate thresholds
                   (if None, uses xarray_data)
    - pooling_window: int or None. If set, inter-event gaps of at most this
                      number of days are filled, merging nearby drought events
                      (inter-event time pooling). If None, no pooling is applied.
    - min_duration: int or None. If set, drought events shorter than this number
                    of days are removed from the drought mask. If None, all events
                    are kept.
    - seasonal_threshold: bool. If True, a daily variable threshold is used:
                          the quantile is computed per day-of-year from a 30-day
                          centred moving average of the threshold series, giving
                          a smooth seasonal envelope. If False (default), a
                          single global threshold is used.

    Returns:
    - DataFrame with drought metrics per gauge.
    """
    
    if threshold_df is None:
        threshold_df = xarray_data
    
    results = []
    
    for gauge_id in xarray_data.gauge_id.values:
        flow_series = xarray_data.sel(gauge_id=gauge_id).to_series()
        threshold_series = threshold_df.sel(gauge_id=gauge_id).to_series()
        
        # 1) Calculate threshold and mask
        is_drought, threshold_val = _calculate_gauge_drought_mask_and_threshold(
            flow_series, threshold_series,
            threshold_quantile=threshold_quantile,
            pooling_window=pooling_window,
            min_duration=min_duration,
            seasonal_threshold=seasonal_threshold
        )
        
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

        ## 5) Count drought events
        events_per_year = is_drought & (~is_drought.shift(1, fill_value=False))
        events_per_year = events_per_year.resample("YE").sum().mean()

        # 5-7) Drought severity and intensity
        drought_values = flow_series.where(is_drought)
        deficit = (threshold_val - drought_values).clip(lower=0)
        severity = float(deficit.sum(skipna=True))
        duration_total = is_drought.sum()
        intensity = float(severity / duration_total) if duration_total > 0 else float("nan")

        results.append({
            'gauge_id': str(gauge_id),
            f'Q90{suffix}': float(Q90),  
            f'annual_min_value{suffix}': float(min_value),
            f'min_30day{suffix}': float(min_30day),
            f'doy_min{suffix}': int(doy_min) if pd.notna(doy_min) else None,
            f'events_per_year{suffix}': float(events_per_year),
            f'severity{suffix}': severity,
            f'total_duration{suffix}': duration_total,
            f'intensity{suffix}': intensity,
        })
    
    df = pd.DataFrame(results).set_index('gauge_id')
    return df

def drought_area_timeseries(xarray_data, metadata, threshold_quantile=0.1, threshold_df=None, pooling_window=None, min_duration=None, seasonal_threshold=False):
    """
    Returns a time series of total area affected by droughts.
    
    Parameters:
    - xarray_data: xarray DataArray with dimensions ("date", "gauge_id")
    - metadata: DataFrame with basin_area indexed by gauge_id
    - threshold_quantile: quantile used for threshold (e.g., 0.1 for Q90)
    - threshold_df: optional xarray DataArray used to calculate thresholds
                   (if None, uses xarray_data)
    - pooling_window: int or None. If set, inter-event gaps of at most this
                      number of days are filled.
    - min_duration: int or None. If set, drought events shorter than this standard
                    duration are removed.
    - seasonal_threshold: bool. If True, a daily variable threshold is used.
    
    Returns:
    - DataFrame with dates as index and latitude as columns, containing drought areas
    """
    if threshold_df is None:
        threshold_df = xarray_data
    
    # Calculate mask per gauge using the exact same logic as find_droughts
    drought_masks = {}
    for gauge_id in xarray_data.gauge_id.values:
        flow_series = xarray_data.sel(gauge_id=gauge_id).to_series()
        threshold_series = threshold_df.sel(gauge_id=gauge_id).to_series()
        is_d_series, _ = _calculate_gauge_drought_mask_and_threshold(
            flow_series, threshold_series,
            threshold_quantile=threshold_quantile,
            pooling_window=pooling_window,
            min_duration=min_duration,
            seasonal_threshold=seasonal_threshold
        )
        drought_masks[gauge_id] = is_d_series
        
    is_drought_df = pd.DataFrame(drought_masks)
    is_drought_df.index.name = 'date'
    
    # Convert to DataFrame for easier manipulation
    drought_df = is_drought_df.reset_index().melt(
        id_vars='date', 
        var_name='gauge_id', 
        value_name='is_drought'
    )
    
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
    
    # Reindex to include all dates with zero fill
    full_date_range = pd.date_range(start=xarray_data.date.min().values, end=xarray_data.date.max().values, freq='D')
    ts = ts.reindex(full_date_range, fill_value=0)
    
    return ts