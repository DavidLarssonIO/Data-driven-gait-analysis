import holoviews as hv


def timeseries_per_skier(df, df_peaks):
    """
    Parameters
    ----------
    test : type
        desc

    Returns
    -------
    test : type
        desc
    """
    force_time = hv.Table(
        df,
        ["Skier"],
        ["Time (sec)", "Force left (N)", "Force right (N)", "Gear"],
    )

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
            "Impulse axial",
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
    curve_force_time = force_time.to.curve(
        "Time (sec)", "Force right (N)"
    ).opts(color="g")
    curve_force_time_l = force_time.to.curve(
        "Time (sec)", "Force left (N)"
    ).opts(color="r")

    plot = (curve_force_time * curve_force_time_l).opts(
        hv.opts.Overlay(height=500, width=1000)
    )
    return plot
