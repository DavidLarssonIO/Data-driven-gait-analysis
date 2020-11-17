import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from bisect import bisect
from sklearn.neighbors import NearestNeighbors


def ImportData(filepath, skier):
    df = pd.read_csv(filepath).dropna()
    df = df.reset_index(drop=True)
    # Time is in min but line 12 converting to sec
    df.columns = [
        "Time (sec)",
        "Force left (N)",
        "Force right (N)",
        "Time (floor min)",
        "Slope (%)",
        "Speed (km/h)",
        "Gear",
    ]
    df.loc[:, "Time (sec)"] = df.loc[:, "Time (sec)"] * 60
    df["Skier"] = [skier] * len(df.index)
    return df


def GetStrokes(
    df_force,
    df_gear,
    df_time,
    df_skier,
    height=25,
    width=1,
    distance=50,
    rel_height=0.93,
    cutoff_start=2,
    cutoff_end=0,
):
    # height=25
    # Updating the cutoff at the end
    cutoff_end = df_time.values[-1] - cutoff_end
    # Getting the peaks from force data
    peaks, _ = find_peaks(df_force, height=height, width=width, distance=distance)
    # Getting the start and stop possition of groundcontact, with interpolation
    _, _, left_ips, right_ips = peak_widths(df_force, peaks, rel_height=rel_height)
    # Rounding to get rid of interpolation
    left_ips = np.floor(left_ips)
    right_ips = np.ceil(right_ips)

    # Getting the time values from index values
    ground_start = df_time[left_ips].values
    ground_stop = df_time[right_ips].values
    # Triming for cutoff
    index_trim = np.logical_and(
        ground_start >= cutoff_start, ground_start <= cutoff_end
    )
    ground_start = ground_start[index_trim]

    # Wanting to get a ground start as first value and a ground_stop as last value.
    index_trim = np.logical_and(
        ground_start[0] < ground_stop, ground_stop < ground_start[-1]
    )
    # Updating with the new trim
    peaks = peaks[index_trim]
    left_ips = left_ips[index_trim]
    right_ips = right_ips[index_trim]
    gear = df_gear[peaks]
    skier = df_skier[peaks]
    # gear = gear[index_trim]
    ground_stop = ground_stop[index_trim]
    force_area = [0] * len(peaks)
    for i in range(len(peaks)):
        start = int(left_ips[i])
        stop = int(right_ips[i])
        force_area[i] = np.trapz(
            df_force[start:stop].values, df_time[start:stop].values
        )
    data = {
        "Gear": gear.values,
        "Peak time": df_time[peaks],
        "Peak height": df_force[peaks],
        "Stroke and ground contact start": ground_start[:-1],
        "Height start": [0] * len(ground_start[:-1]),
        "Ground contact stop": ground_stop,
        "Height stop": [0] * len(ground_stop),
        "Stroke stop": ground_start[1:],
        "Skier": skier,
        "Force area": force_area,
    }
    df_peaks = pd.DataFrame(data)
    # Deleate last value
    df_peaks.drop(df_peaks.tail(1).index, inplace=True)
    return df_peaks


def GetInfo(df_peaks):
    stroke_time = (
        df_peaks["Stroke stop"].values
        - df_peaks["Stroke and ground contact start"].values
    )
    ground_contact_time = (
        df_peaks["Ground contact stop"].values
        - df_peaks["Stroke and ground contact start"].values
    )
    air_time = df_peaks["Stroke stop"].values - df_peaks["Ground contact stop"].values

    data = {
        "Gear": df_peaks["Gear"].values,
        "Stroke time": stroke_time,
        "Ground contact time": ground_contact_time,
        "Air time": air_time,
        "Frequency": 1 / stroke_time,
        "Force area": df_peaks["Force area"].values,
        "Skier": df_peaks["Skier"].values,
        "Peak time": df_peaks["Peak time"],
    }
    df_info = pd.DataFrame(data)
    return df_info


def generate_stroke_dataframe(filepath, i):
    df = ImportData(filepath, i)
    df_peaks_left = GetStrokes(
        df["Force left (N)"], df["Gear"], df["Time (sec)"], df["Skier"]
    )
    df_peaks_right = GetStrokes(
        df["Force right (N)"], df["Gear"], df["Time (sec)"], df["Skier"]
    )
    df_info_left = GetInfo(df_peaks_left)
    df_info_right = GetInfo(df_peaks_right)
    df_info_left, df_info_right = add_left_and_right_diff(
        df_peaks_left, df_peaks_right, df_info_left, df_info_right
    )
    return df, df_peaks_left, df_peaks_right, df_info_left, df_info_right


def add_left_and_right_diff(df_peaks_left, df_peaks_right, df_info_left, df_info_right):
    nbrs_left = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(
        np.transpose([df_peaks_left["Peak time"].values])
    )
    nbrs_right = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(
        np.transpose([df_peaks_right["Peak time"].values])
    )
    times_right, _ = nbrs_left.kneighbors(
        np.transpose([df_peaks_right["Peak time"].values])
    )
    times_left, _ = nbrs_right.kneighbors(
        np.transpose([df_peaks_left["Peak time"].values])
    )
    df_info_left["Time to other pole"] = times_left
    df_info_right["Time to other pole"] = times_right
    return df_info_left, df_info_right
