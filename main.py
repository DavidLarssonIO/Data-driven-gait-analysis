import pandas as pd
from functions.plotting_functions import timeseries_per_skier
import holoviews as hv

hv.extension("bokeh")


df = pd.read_pickle("./data/timeseries.pkl")
df_peaks = pd.read_pickle("./data/peaks.pkl")

plot = timeseries_per_skier(df, df_peaks[df_peaks["Pole"] == "Right"])
hv.save(plot, "figures/Timeseries.html")
