import matplotlib.pyplot as plt


def plot_contact_detection(df, title, start_time=20, end_time=30):
    """
    Plot left/right force signals together with detected contact markers.
    """
    zoom_df = df[(df["time"] >= start_time) & (df["time"] <= end_time)]

    plt.figure(figsize=(14, 5))

    plt.plot(
        zoom_df["time"],
        zoom_df["left_total_force"],
        label="Left total force"
    )

    plt.plot(
        zoom_df["time"],
        zoom_df["right_total_force"],
        label="Right total force"
    )

    plt.scatter(
        zoom_df["time"],
        zoom_df["left_contact"] * 50,
        s=5,
        label="Left contact detected"
    )

    plt.scatter(
        zoom_df["time"],
        zoom_df["right_contact"] * 100,
        s=5,
        label="Right contact detected"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title(title)
    plt.legend()
    plt.show()