import pandas as pd


def add_foot_contact_columns(df, threshold=20):
    """
    Add Boolean columns showing whether left/right foot is in contact with the ground.
    """
    df = df.copy()

    df["left_contact"] = df["left_total_force"] > threshold
    df["right_contact"] = df["right_total_force"] > threshold

    return df


def detect_contact_events(df, contact_column):
    """
    Detect continuous contact periods from a Boolean contact column.
    """
    contact = df[contact_column].astype(bool)

    contact_start = contact & ~contact.shift(1, fill_value=False)
    contact_end = contact & ~contact.shift(-1, fill_value=False)

    start_indices = df.index[contact_start].to_list()
    end_indices = df.index[contact_end].to_list()

    events = []

    for start_idx, end_idx in zip(start_indices, end_indices):
        start_time = df.loc[start_idx, "time"]
        end_time = df.loc[end_idx, "time"]
        duration = end_time - start_time

        events.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration
        })

    return pd.DataFrame(events)


def filter_contact_events(events, min_duration=0.2, max_duration=2.0):
    """
    Remove unrealistically short or long contact events.
    """
    if events.empty:
        return events

    return events[
        (events["duration"] >= min_duration) &
        (events["duration"] <= max_duration)
    ].reset_index(drop=True)