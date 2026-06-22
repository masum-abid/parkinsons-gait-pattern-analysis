import numpy as np
import pandas as pd

from contact_detection import (
    add_foot_contact_columns,
    detect_contact_events,
    filter_contact_events
)


def safe_mean(series):
    return series.mean() if len(series) > 0 else np.nan


def safe_std(series):
    return series.std() if len(series) > 1 else np.nan


def safe_cv(series):
    mean_value = safe_mean(series)
    std_value = safe_std(series)

    if pd.isna(mean_value) or mean_value == 0:
        return np.nan

    return std_value / mean_value


def extract_contact_features(df, threshold=20):
    """
    Extract basic contact-based gait features from one recording.
    """
    df = add_foot_contact_columns(df, threshold=threshold)

    left_events = detect_contact_events(df, "left_contact")
    right_events = detect_contact_events(df, "right_contact")

    left_events_clean = filter_contact_events(left_events)
    right_events_clean = filter_contact_events(right_events)

    recording_duration = df["time"].max() - df["time"].min()

    features = {
        "recording_duration_sec": recording_duration,

        "left_contact_count": len(left_events_clean),
        "right_contact_count": len(right_events_clean),

        "left_contacts_per_sec": len(left_events_clean) / recording_duration,
        "right_contacts_per_sec": len(right_events_clean) / recording_duration,

        "mean_left_contact_duration": safe_mean(left_events_clean["duration"]),
        "mean_right_contact_duration": safe_mean(right_events_clean["duration"]),

        "std_left_contact_duration": safe_std(left_events_clean["duration"]),
        "std_right_contact_duration": safe_std(right_events_clean["duration"]),

        "cv_left_contact_duration": safe_cv(left_events_clean["duration"]),
        "cv_right_contact_duration": safe_cv(right_events_clean["duration"]),

        "contact_count_difference": abs(
            len(left_events_clean) - len(right_events_clean)
        ),

        "mean_contact_duration_difference": abs(
            safe_mean(left_events_clean["duration"]) -
            safe_mean(right_events_clean["duration"])
        ),

        "cv_contact_duration_difference": abs(
            safe_cv(left_events_clean["duration"]) -
            safe_cv(right_events_clean["duration"])
        ),
    }

    return features

def add_force_features_to_events(df, events, force_column):
    events = events.copy()
    
    peak_forces = []
    mean_forces = []
    force_areas = []
    
    for _, event in events.iterrows():
        start_idx = int(event["start_idx"])
        end_idx = int(event["end_idx"])
        
        event_force = df.loc[start_idx:end_idx, force_column]
        event_time = df.loc[start_idx:end_idx, "time"]
        
        peak_forces.append(event_force.max())
        mean_forces.append(event_force.mean())
        
        if len(event_force) > 1:
            force_area = np.trapz(event_force, event_time)
        else:
            force_area = np.nan
            
        force_areas.append(force_area)
    
    events["peak_force"] = peak_forces
    events["mean_force"] = mean_forces
    events["force_area"] = force_areas
    
    return events

def compute_stride_times(events):
    if len(events) < 2:
        return pd.Series(dtype=float)
    
    return events["start_time"].diff().dropna()

def compute_swing_times(events):
    if len(events) < 2:
        return pd.Series(dtype=float)
    
    swing_times = events["start_time"].iloc[1:].values - events["end_time"].iloc[:-1].values
    return pd.Series(swing_times)

def compute_step_times(left_events, right_events):
    event_times = []
    
    for t in left_events["start_time"]:
        event_times.append({"time": t, "foot": "left"})
        
    for t in right_events["start_time"]:
        event_times.append({"time": t, "foot": "right"})
    
    if len(event_times) < 2:
        return pd.Series(dtype=float)
    
    event_df = pd.DataFrame(event_times).sort_values("time")
    
    # Keep only alternating foot events
    event_df["previous_foot"] = event_df["foot"].shift(1)
    event_df["is_alternating"] = event_df["foot"] != event_df["previous_foot"]
    
    alternating_events = event_df[event_df["is_alternating"]].copy()
    
    step_times = alternating_events["time"].diff().dropna()
    
    return step_times

def symmetry_index(left_value, right_value):
    if pd.isna(left_value) or pd.isna(right_value):
        return np.nan
    
    denominator = (left_value + right_value) / 2
    
    if denominator == 0:
        return np.nan
    
    return abs(left_value - right_value) / denominator

def extract_improved_gait_features(df, threshold=20):
    df = add_foot_contact_columns(df, threshold=threshold)
    
    left_events = detect_contact_events(df, "left_contact")
    right_events = detect_contact_events(df, "right_contact")
    
    left_events = filter_contact_events(left_events)
    right_events = filter_contact_events(right_events)
    
    left_events = add_force_features_to_events(
        df,
        left_events,
        "left_total_force"
    )
    
    right_events = add_force_features_to_events(
        df,
        right_events,
        "right_total_force"
    )
    
    left_stride_times = compute_stride_times(left_events)
    right_stride_times = compute_stride_times(right_events)
    
    left_swing_times = compute_swing_times(left_events)
    right_swing_times = compute_swing_times(right_events)
    
    step_times = compute_step_times(left_events, right_events)
    
    # Old/basic contact features
    basic_features = extract_contact_features(df, threshold=threshold)
    
    # New features
    features = {
        **basic_features,
        
        # Step timing
        "mean_step_time": safe_mean(step_times),
        "std_step_time": safe_std(step_times),
        "cv_step_time": safe_cv(step_times),
        "median_step_time": safe_median(step_times),
        
        # Stride timing
        "mean_left_stride_time": safe_mean(left_stride_times),
        "mean_right_stride_time": safe_mean(right_stride_times),
        "std_left_stride_time": safe_std(left_stride_times),
        "std_right_stride_time": safe_std(right_stride_times),
        "cv_left_stride_time": safe_cv(left_stride_times),
        "cv_right_stride_time": safe_cv(right_stride_times),
        
        # Swing timing
        "mean_left_swing_time": safe_mean(left_swing_times),
        "mean_right_swing_time": safe_mean(right_swing_times),
        "std_left_swing_time": safe_std(left_swing_times),
        "std_right_swing_time": safe_std(right_swing_times),
        "cv_left_swing_time": safe_cv(left_swing_times),
        "cv_right_swing_time": safe_cv(right_swing_times),
        
        # Peak force
        "mean_left_peak_force": safe_mean(left_events["peak_force"]),
        "mean_right_peak_force": safe_mean(right_events["peak_force"]),
        "std_left_peak_force": safe_std(left_events["peak_force"]),
        "std_right_peak_force": safe_std(right_events["peak_force"]),
        "cv_left_peak_force": safe_cv(left_events["peak_force"]),
        "cv_right_peak_force": safe_cv(right_events["peak_force"]),
        
        # Force area
        "mean_left_force_area": safe_mean(left_events["force_area"]),
        "mean_right_force_area": safe_mean(right_events["force_area"]),
        "cv_left_force_area": safe_cv(left_events["force_area"]),
        "cv_right_force_area": safe_cv(right_events["force_area"]),
        
        # Asymmetry
        "stride_time_symmetry_index": symmetry_index(
            safe_mean(left_stride_times),
            safe_mean(right_stride_times)
        ),
        "swing_time_symmetry_index": symmetry_index(
            safe_mean(left_swing_times),
            safe_mean(right_swing_times)
        ),
        "peak_force_symmetry_index": symmetry_index(
            safe_mean(left_events["peak_force"]),
            safe_mean(right_events["peak_force"])
        ),
        "force_area_symmetry_index": symmetry_index(
            safe_mean(left_events["force_area"]),
            safe_mean(right_events["force_area"])
        ),
    }
    
    return features