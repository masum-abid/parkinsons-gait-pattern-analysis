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