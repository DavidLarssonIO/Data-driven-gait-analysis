from functions.dataframe_functions import get_dataframe
from functions.plotting_functions import timeseries_per_skier
import holoviews as hv
hv.extension("bokeh")

path_first = "../matlab_report-master/Treadmill/Test/Torsby/"
skier_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
path_second = "/CSV/csvData.csv"
filepath_list = [path_first + str(i) + path_second for i in skier_list]

df, df_peaks, _ = get_dataframe(filepath_list, skier_list)

plot = timeseries_per_skier(df, df_peaks[df_peaks["Pole"] == 1])
hv.save(plot, "figures/Timeseries.html")
