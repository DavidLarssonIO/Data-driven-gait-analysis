import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
from holoviews.operation import gridmatrix
hv.extension('bokeh')

from functions.dataframe_functions import *
from functions.plotting_functions import *

path_first = "../matlab_report-master/Treadmill/Test/Torsby/"
path_int = [1,2,3]#,4,5,6,7,8]
path_second = "/CSV/csvData.csv"
for i in path_int:
    filepath = path_first+str(i)+path_second
    if i == path_int[0]:
        df, df_peaks_left, df_peaks_right, df_info_left, df_info_right = generate_stroke_dataframe(filepath,i)
    else:
        df_tmp, df_peaks_left_tmp, df_peaks_right_tmp, df_info_left_tmp, df_info_right_tmp = generate_stroke_dataframe(filepath,i)
        df = df.append(df_tmp, ignore_index=True)
        df_peaks_left = df_peaks_left.append(df_peaks_left_tmp, ignore_index=True)
        df_peaks_right = df_peaks_right.append(df_peaks_right_tmp, ignore_index=True)
        df_info_left = df_info_left.append(df_info_left_tmp, ignore_index=True)
        df_info_right = df_info_right.append(df_info_right_tmp, ignore_index=True)

df_sub = df_info_left[["Gear","Stroke time","Ground contact time","Air time","Frequency","Force area","Time to other pole"]]
ds = hv.Dataset(df_sub).groupby('Gear').overlay()
point_grid = gridmatrix(ds, diagonal_type=hv.Distribution, chart_type=hv.Points)
point_grid.opts(opts.Points(size=2, alpha=0.5),opts.NdOverlay(batched=False))
hv.save(point_grid, 'Points.html')
