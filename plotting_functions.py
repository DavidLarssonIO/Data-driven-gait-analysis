import holoviews as hv


def timeseries_per_skier(df,df_peaks):
    force_time = hv.Table(df, 
                          ["Time (sec)","Force left (N)","Skier"], 
                          ['Force right (N)', 'Time (floor min)','Slope (%)', 'Speed (km/h)', 'Gear'])
    peaks = hv.Table(df_peaks,
                     ["Skier"],
                     ["Gear",
                      "Peak time",
                      "Peak height",
                      "Stroke and ground contact start",
                      "Height start",
                      "Height stop",
                      "Ground contact stop",
                      "Stroke stop",
                      "Impulse"])
    scatter_peaks = peaks.to.scatter('Peak time', 'Peak height').opts(color='k', marker='^', size=5)
    scatter_start = peaks.to.scatter("Stroke and ground contact start", 'Height start').opts(color='g', marker='>', size=5)
    scatter_stop = peaks.to.scatter("Ground contact stop", 'Height stop').opts(color='r', marker='<', size=5)
    curv_force_time = force_time.to.curve('Time (sec)', 'Force left (N)')
    plot = (curv_force_time*scatter_peaks*scatter_start*scatter_stop).opts(hv.opts.Overlay(height=500,width=1000))
    return plot
