import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from sklearn.neighbors import NearestNeighbors


def get_dataframe(filepath_list, skier_list):
    for j in range(len(skier_list)):
        i = skier_list[j]
        filepath = filepath_list[j]
        if i == skier_list[0]:
            df, df_peaks_left, df_peaks_right, df_info_left, df_info_right = generate_stroke_dataframe(
                filepath, i
            )
            df_peaks = df_peaks_left
            df_peaks = df_peaks.append(df_peaks_right, ignore_index=True)
            df_info = df_info_left
            df_info = df_info.append(df_info_right, ignore_index=True)
        else:
            df_tmp, df_peaks_left_tmp, df_peaks_right_tmp, df_info_left_tmp, df_info_right_tmp = generate_stroke_dataframe(
                filepath, i
            )
            df = df.append(df_tmp, ignore_index=True)
            df_peaks = df_peaks.append(df_peaks_left_tmp, ignore_index=True)
            df_peaks = df_peaks.append(df_peaks_right_tmp, ignore_index=True)
            df_info = df_info.append(df_info_left_tmp, ignore_index=True)
            df_info = df_info.append(df_info_right_tmp, ignore_index=True)
    return df, df_peaks, df_info


def generate_stroke_dataframe(filepath, i):
    df = ImportData(filepath, i)

    df["Time (sec)"] = np.linspace(
        np.min(df["Time (sec)"]), np.max(df["Time (sec)"]), len(df.index)
    )
    df["Time (min)"] = np.linspace(
        np.min(df["Time (min)"]), np.max(df["Time (min)"]), len(df.index)
    )

    int_add = np.sum(df["Time (sec)"].values <= 12)
    df["Gear"] = df["Gear"].shift(periods=int_add, fill_value=df["Gear"][0])

    df_peaks_left = GetStrokes(
        df["Force left (N)"], df["Gear"], df["Time (sec)"], df["Skier"], 0
    )
    df_peaks_right = GetStrokes(
        df["Force right (N)"], df["Gear"], df["Time (sec)"], df["Skier"], 1
    )
    df_info_left = GetInfo(df_peaks_left)
    df_info_right = GetInfo(df_peaks_right)
    df_info_left, df_info_right = add_left_and_right_diff(
        df_peaks_left, df_peaks_right, df_info_left, df_info_right
    )
    return df, df_peaks_left, df_peaks_right, df_info_left, df_info_right


def ImportData(filepath, skier):
    df = pd.read_csv(filepath).dropna()
    df = df.reset_index(drop=True)
    df.columns = [
        "Time (min)",
        "Force left (N)",
        "Force right (N)",
        "Time (floor min)",
        "Slope (%)",
        "Speed (km/h)",
        "Gear",
    ]
    df["Skier"] = [skier] * len(df.index)
    df["Time (sec)"] = df["Time (min)"] * 60
    trim = (df["Speed (km/h)"].values == 0) * (df["Time (sec)"].values > 800)

    df = df[~trim]
    return df


def GetStrokes(
    df_force,
    df_gear,
    df_time,
    df_skier,
    pole,
    height=0.1,
    width=1,
    distance=64,
    rel_height=0.97,
    cutoff_start=2,
    cutoff_end=2,
):
    height = height * np.max(df_force)
    # Updating the cutoff at the end
    cutoff_end = df_time.values[-1] - cutoff_end
    # Getting the peaks from force data
    peaks, _ = find_peaks(
        df_force, height=height, width=width, distance=distance
    )
    # Getting the start and stop possition of groundcontact, with interpolation
    _, _, left_ips, right_ips = peak_widths(
        df_force, peaks, rel_height=rel_height
    )
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
        "Pole": pole,
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
    air_time = (
        df_peaks["Stroke stop"].values - df_peaks["Ground contact stop"].values
    )

    data = {
        "Gear": df_peaks["Gear"].values,
        "Stroke time": stroke_time,
        "Pull time": ground_contact_time,
        "Recovery time": air_time,
        "Frequency": 1 / stroke_time,
        "Impulse axial": df_peaks["Force area"].values,
        "Skier": df_peaks["Skier"].values,
        "Peak time": df_peaks["Peak time"].values,
        "Pole": df_peaks["Pole"].values,
    }
    df_info = pd.DataFrame(data)
    return df_info


def add_left_and_right_diff(
    df_peaks_left, df_peaks_right, df_info_left, df_info_right
):
    nbrs_left = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(
        np.transpose([df_peaks_left["Peak time"].values])
    )
    nbrs_right = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(
        np.transpose([df_peaks_right["Peak time"].values])
    )
    times_right, indices_right = nbrs_left.kneighbors(
        np.transpose([df_peaks_right["Peak time"].values])
    )
    times_left, indices_left = nbrs_right.kneighbors(
        np.transpose([df_peaks_left["Peak time"].values])
    )
    df_info_left["Time to other pole"] = (
        np.transpose(times_left)[0] / df_info_left["Stroke time"].values
    )
    df_info_left["Other pole index"] = indices_left
    df_info_right["Other pole index"] = indices_right
    df_info_right["Time to other pole"] = (
        np.transpose(times_right)[0] / df_info_right["Stroke time"].values
    )
    height_left = df_peaks_left["Peak height"].reset_index(drop=True)
    height_right = df_peaks_right["Peak height"].reset_index(drop=True)
    height_coresponding_left = height_right.iloc[indices_left.flatten()]
    height_coresponding_right = height_left.iloc[indices_right.flatten()]
    df_info_left["Peak height diff"] = np.abs(
        height_coresponding_left.values - height_left.values
    ) / np.max([height_coresponding_left.values, height_left.values])
    df_info_right["Peak height diff"] = np.abs(
        height_coresponding_right.values - height_right.values
    ) / np.max([height_coresponding_right.values, height_right.values])

    return df_info_left, df_info_right


def trim_dataframe(
    df,
    trim_features=[
        "Stroke time",
        "Pull time",
        "Recovery time",
        "Frequency",
        "Impulse axial",
        "Time to other pole",
    ],
    number_of_only_up=1,
    quantile=0.001,
):
    up = [0] * len(trim_features)
    down = [0] * len(trim_features)
    for i in range(len(trim_features)):
        up[i] = df[trim_features[i]].quantile(1 - quantile)
        down[i] = df[trim_features[i]].quantile(quantile)

    for i in range(len(trim_features)):
        if i != len(trim_features) - number_of_only_up:
            df = df[df[trim_features[i]] > down[i]]
        df = df[df[trim_features[i]] < up[i]]
    return df
