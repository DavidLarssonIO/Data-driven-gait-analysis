import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
from holoviews.operation import gridmatrix

hv.extension("bokeh")
from bokeh.models import HoverTool

from functions.dataframe_functions import *
from functions.plotting_functions import *

path_first = "../matlab_report-master/Treadmill/Test/Torsby/"
skier_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
path_second = "/CSV/csvData.csv"
filepath_list = [path_first + str(i) + path_second for i in skier_list]

df, df_peaks, df_info = get_dataframe(filepath_list, skier_list)
# df_info["Gear"] = df_info["Gear"].astype('str')
df_sub = df_info[
    [
        "Gear",
        "Stroke time",
        "Ground contact time",
        "Air time",
        "Frequency",
        "Force area",
        "Time to other pole",
    ]
]
print(df_sub.describe())
fix = [
    "Stroke time",
    "Ground contact time",
    "Air time",
    "Frequency",
    "Force area",
    "Time to other pole",
]
up = [0] * len(fix)
down = [0] * len(fix)
for i in range(len(fix)):
    up[i] = df_sub[fix[i]].quantile(0.999)
    down[i] = df_sub[fix[i]].quantile(0.001)

for i in range(len(fix)):
    if i != len(fix) - 1:
        df_sub = df_sub[df_sub[fix[i]] > down[i]]
    df_sub = df_sub[df_sub[fix[i]] < up[i]]
print(df_sub.describe())
df_sub = df_sub.sample(7000)
ds = hv.Dataset(df_sub).groupby("Gear").overlay()

point_grid = gridmatrix(ds, diagonal_type=hv.Distribution, chart_type=hv.Points)
point_grid.opts(opts.Points(size=2, alpha=0.3, tools=["hover", "box_select"]))
legend = point_grid[("Air time", "Force area")].options(
    xaxis=None, yaxis=None, legend_position="top_left", show_frame=False, width=150
)
legend.opts(opts.Points(size=0))
hv.save(point_grid + legend, "figures/Points.html")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipline = Pipeline([("scaling", StandardScaler()), ("pca", PCA(n_components=2))])
pca = PCA(n_components=2)
pca_df = df_sub[
    [
        "Stroke time",
        "Ground contact time",
        "Air time",
        "Frequency",
        "Force area",
        "Time to other pole",
        "Gear",
    ]
].sample(400)
gear = pca_df["Gear"].astype("str")
pca_df = pca_df[
    [
        "Time to other pole",
        "Stroke time",
        "Ground contact time",
        "Air time",
        "Frequency",
        "Force area",
        "Time to other pole",
    ]
]
principalComponents = pipline.fit_transform(pca_df.values)
principalDf = pd.DataFrame(
    data=principalComponents, columns=["Principal component 1", "Principal component 2"]
)
gear = pd.DataFrame(gear.values, columns=["Gear"])
finalDf = pd.concat([principalDf, gear], axis=1)
points = hv.Points(
    finalDf, ["Principal component 1", "Principal component 2"], ["Gear"]
).sort("Gear")
points = points.opts(
    color="Gear", height=500, width=500, size=5, cmap="Category20", aspect="equal"
)

hv.save(points, "figures/PCA.html")
overlay = hv.Layout(point_grid.relabel("Gridplot") + points.relabel("PCA")).opts(
    tabs=True
)
hv.save(overlay, "figures/Dashboard.html")
