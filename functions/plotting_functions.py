import holoviews as hv


def timeseries_per_skier(df, df_peaks):
    # df["Force left (N)"] = df["Force left (N)"].shift(periods=60, fill_value=0)
    force_time = hv.Table(
        df,
        ["Skier"],
        [
            "Time (sec)",
            "Force left (N)",
            "Force right (N)",
            "Gear",
        ],  # "Time (floor min)", "Slope (%)", "Speed (km/h)", "Gear"],
    )
    # df_peaks.loc[:, "Peak time"] = df_peaks.loc[:, "Peak time"] / 60
    # df_peaks.loc[:, "Stroke and ground contact start"] = (
    #    df_peaks.loc[:, "Stroke and ground contact start"] / 60
    # )
    # df_peaks.loc[:, "Ground contact stop"] = df_peaks.loc[:, "Ground contact stop"] / 60
    peaks = hv.Table(
        df_peaks,
        ["Skier"],
        [
            "Gear",
            "Peak time",
            "Peak height",
            "Stroke and ground contact start",
            "Height start",
            "Height stop",
            "Ground contact stop",
            "Stroke stop",
            "Force area",
        ],
    )
    scatter_peaks = peaks.to.scatter("Peak time", "Peak height").opts(
        color="k", marker="^", size=5
    )
    scatter_start = peaks.to.scatter(
        "Stroke and ground contact start", "Height start"
    ).opts(color="g", marker=">", size=5)
    scatter_stop = peaks.to.scatter("Ground contact stop", "Height stop").opts(
        color="r", marker="<", size=5
    )
    curve_force_time = force_time.to.curve("Time (sec)", "Force right (N)").opts(
        color="g"
    )
    curve_force_time_l = force_time.to.curve("Time (sec)", "Force left (N)").opts(
        color="r"
    )
    # path_force_time_right = hv.Path([(df['Time (sec)'],df["Force right (N)"], df["Gear"])], vdims='Gear').opts(color='Gear', colorbar=True)
    plot = (
        curve_force_time
        * curve_force_time_l
        # * scatter_peaks
        # * scatter_start
        # * scatter_stop
    ).opts(hv.opts.Overlay(height=500, width=1000))
    return plot
