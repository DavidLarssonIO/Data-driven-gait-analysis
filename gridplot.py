import pandas as pd
import holoviews as hv
from holoviews import opts
from holoviews.operation import gridmatrix
import seaborn as sns
from functions.dataframe_functions import delete_outlier

hv.extension("bokeh")


df_info = pd.read_pickle("./data/strokes.pkl")

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
# if len(df_sub.index) > 7000:
#    df_sub = df_sub.sample(7000)
ds = hv.Dataset(df_sub).groupby("Gear").overlay()
point_grid = gridmatrix(ds, chart_type=hv.Points)
point_grid.opts(opts.Points(size=1, alpha=0.3, cmap="Plasma"))
legend = point_grid[("Frequency", "Time to other pole")].opts(
    xaxis=None,
    yaxis=None,
    legend_position="top_left",
    show_frame=False,
    width=150,
)
# legend = point_grid[("Frequency", "Impulse axial")].opts(
# legend.opts(opts.Points(size=0, alpha=1))
# hv.save(point_grid + legend, "figures/Points.html")
df_sub = delete_outlier(df_sub)
sns_plot = sns.pairplot(
    df_sub,
    hue="Gear",
    diag_kind="hist",
    corner=True,
    palette="flare",
    plot_kws={"s": 2},
    height=1.5,
)
sns_plot.savefig("figures/Gait_analysis.png")
