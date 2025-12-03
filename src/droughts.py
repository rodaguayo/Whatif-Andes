"""Module for identifying drought events in time series data."""
import pandas as pd
from tqdm import tqdm

def find_droughts(df, quantile, threshold_df=None):
    """
    Identifies drought events in a time series based on a quantile threshold.

    Parameters:
    df (pd.DataFrame): A Pandas DataFrame with a datetime index and streamflow
                       values in columns, where each column represents a gauge.
                       This DataFrame is used to identify the events.
    quantile (float): The quantile (e.g., 0.1 for 10th percentile) to
                      determine the threshold.
    threshold_df (pd.DataFrame, optional): A DataFrame with a similar structure
                                           to `df`, used for calculating the
                                           quantile threshold. If None, `df` is
                                           used for both event identification
                                           and threshold calculation.

    Returns:
    pd.DataFrame: A DataFrame where each row represents a drought event from
                  any gauge, with columns for start date, end date, duration,
                  severity, min_value, and gauge_id.
    """
    df.index = pd.to_datetime(df.index)
    if threshold_df is None:
        threshold_df = df
    else:
        threshold_df.index = pd.to_datetime(threshold_df.index)
        
    all_events = []

    for gauge_id in tqdm(df.columns):
        
        series = df[gauge_id]
        threshold_series = threshold_df[gauge_id]
        threshold = threshold_series.quantile(quantile)
        below_threshold = series < threshold
        
        # Detect start and end of drought events
        drought_starts = below_threshold & ~below_threshold.shift(1, fill_value=False)
        drought_ends = ~below_threshold & below_threshold.shift(1, fill_value=False)

        start_dates = series.index[drought_starts]
        end_dates = series.index[drought_ends]

        # Handle open-ended droughts (if the series ends during a drought)
        if len(start_dates) > len(end_dates):
            end_dates = end_dates.append(pd.Index([series.index[-1]]))
        
        # Handle cases where a drought starts but doesn't end within the series
        if len(start_dates) < len(end_dates):
            start_dates = start_dates.insert(0, series.index[0])


        gauge_events = []
        for start, end in zip(start_dates, end_dates):
            # Slice the series for the event
            event_values = series[start:end]
            if event_values.empty:
                continue
            duration = (end - start).days + 1
            severity = (threshold - event_values[event_values < threshold]).sum()
            min_value = event_values.min()

            gauge_events.append({
                "gauge_id": gauge_id,
                "start_date": start,
                "end_date": end,
                "duration": duration,
                "severity": severity,
                "min_value": min_value
            })
        
        if gauge_events:
            all_events.extend(gauge_events)

    # Create a DataFrame with all events
    return pd.DataFrame(all_events)


def drought_area_timeseries(droughts_ref, metadata):
    """
    Returns a time series of total area affected by droughts.
    """
    # Create a DataFrame with all drought periods for each basin
    records = []
    for _, row in tqdm(droughts_ref.iterrows(), total=droughts_ref.shape[0]):
        gauge_id = row['gauge_id']
        area = metadata.loc[gauge_id, 'basin_area']
        # Create a date range for the drought period
        dates = pd.date_range(row['start_date'], row['end_date'])
        for date in dates:
            records.append({'date': date, 'gauge_id': gauge_id, 'area': area})
    df_area = pd.DataFrame(records)
    # Group by date and sum the area
    ts = df_area.groupby('date')['area'].sum()
    return ts