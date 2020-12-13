import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
from holoviews.operation import gridmatrix
import plotly.express as px

hv.extension("bokeh")
from bokeh.models import HoverTool

from functions.dataframe_functions import *
from functions.plotting_functions import *

df_info = pd.read_pickle("./data/strokes.pkl")

# df_info["Gear"] = df_info["Gear"].astype('str')
df_sub = df_info[
    [
        "Gear",
        "Stroke time",
        "Pull time",
        "Recovery time",
        "Frequency",
        "Impulse axial",
        "Time to other pole",
        "Peak height diff",
    ]
]
df_sub = trim_dataframe(df_sub)
if len(df_sub.index) > 7000:
    df_sub = df_sub.sample(7000)
ds = hv.Dataset(df_sub).groupby("Gear").overlay()
point_grid = gridmatrix(ds, chart_type=hv.Points)  # diagonal_type=hv.Distribution
point_grid.opts(opts.Points(size=1, alpha=0.3, cmap="Plasma"))
legend = point_grid[("Frequency", "Time to other pole")].opts(
    xaxis=None, yaxis=None, legend_position="top_left", show_frame=False, width=150
)
# legend = point_grid[("Frequency", "Impulse axial")].opts(
# legend.opts(opts.Points(size=0, alpha=1))
hv.save(point_grid + legend, "figures/Points.html")
