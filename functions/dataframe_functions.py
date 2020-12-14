import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from sklearn.neighbors import NearestNeighbors


def get_dataframe(filepath_list, skier_list):
    """The main function to call to generate all dataframes
    Parameters
    ----------
    filepath_list : list of strings
        A list of the filepath for the CSV files for the source files
    skier_list : list of ints
        A list of the indentity of the different skiers.

    Returns
    -------
    df : dataframe
        Dataframe over the timeseries data
    df_peaks : dataframe
        Dataframe over the stroke information on a timeseries basis
    df_info : dataframe
        Dataframe over individual pulling, the dataframe for doing
        machine learning since it doesn't contain any timeseries information
    """
    for j in range(len(skier_list)):
        i = skier_list[j]
        filepath = filepath_list[j]
        # If it's the first call we save to variables without append, else with
        # append
        if i == skier_list[0]:
            # Calling the generate_stroke_dataframe to get all of the data
            dataframe_dict = generate_stroke_dataframe(filepath, i)
            # Saving the data to variables
            df = dataframe_dict["df"]
            df_peaks = dataframe_dict["df_peaks_left"]
            df_peaks = df_peaks.append(
                dataframe_dict["df_peaks_right"], ignore_index=True
            )
            df_info = dataframe_dict["df_info_left"]
            df_info = df_info.append(
                dataframe_dict["df_info_right"], ignore_index=True
            )
        else:
            # Calling generate_stroke_dataframe for the rest of the skiers
            dataframe_dict = generate_stroke_dataframe(filepath, i)
            # Appending the data to variables
            df = df.append(dataframe_dict["df"], ignore_index=True)
            df_peaks = df_peaks.append(
                dataframe_dict["df_peaks_left"], ignore_index=True
            )
            df_peaks = df_peaks.append(
                dataframe_dict["df_peaks_right"], ignore_index=True
            )
            df_info = df_info.append(
                dataframe_dict["df_info_left"], ignore_index=True
            )
            df_info = df_info.append(
                dataframe_dict["df_info_right"], ignore_index=True
            )
    return df, df_peaks, df_info


def generate_stroke_dataframe(filepath, i):
    """ Function to call when getting data from one skier
    Parameters
    ----------
    filepath : str
        Filepath for the specific skier
    i : int
        Skier indentity
    Returns
    -------
    dataframe_dict : dict
        A dictionary with keys "df", "df_peaks_left", "df_peaks_right",
        "df_info_left", "df_info_right" which are all dataframes
    """
    # Reading CSV
    df = ImportData(filepath, i)
    # Needing to fix the timeseries, using the same method as MATLAB
    df["Time (sec)"] = np.linspace(
        np.min(df["Time (sec)"]), np.max(df["Time (sec)"]), len(df.index)
    )
    # Same for min
    df["Time (min)"] = np.linspace(
        np.min(df["Time (min)"]), np.max(df["Time (min)"]), len(df.index)
    )

    # Calling GetStrokes and GetInfo for both left and right
    df_peaks_left = GetStrokes(df, "Left")
    df_peaks_right = GetStrokes(df, "Right")
    df_info_left = GetInfo(df_peaks_left)
    df_info_right = GetInfo(df_peaks_right)
    # Adding the time between nearest left and right pulling
    df_info_left, df_info_right = add_left_and_right_diff(
        df_peaks_left, df_peaks_right, df_info_left, df_info_right
    )
    left_0 = df_peaks_left[df_peaks_left["Gear"] == 0][10:90]
    left_0 = left_0["Peak time"].values
    right_0 = df_info_left[df_info_left["Gear"] == 0][10:90]
    right_0 = right_0["Other pole time"].values
    x = left_0
    y = left_0 - right_0
    k, m = np.polyfit(x, y, 1)
    time = df["Time (sec)"].values
    comp = k * np.linspace(np.min(time), np.max(time), len(time)) + m
    time = time - comp
    df["Force right (N)"] = np.interp(
        time, df["Time (sec)"], df["Force right (N)"], left=0, right=0
    )
    df["Time (sec)"] = time
    df["Time (min)"] = time / 60

    df_peaks_left = GetStrokes(df, "Left")
    df_peaks_right = GetStrokes(df, "Right")

    df_info_left = GetInfo(df_peaks_left)
    df_info_right = GetInfo(df_peaks_right)

    # Adding the time between nearest left and right pulling
    df_info_left, df_info_right = add_left_and_right_diff(
        df_peaks_left, df_peaks_right, df_info_left, df_info_right
    )
    # Saving it all to a dictionary
    dataframe_dict = {
        "df": df,
        "df_peaks_left": df_peaks_left,
        "df_peaks_right": df_peaks_right,
        "df_info_left": df_info_left,
        "df_info_right": df_info_right,
    }
    return dataframe_dict


def ImportData(filepath, skier):
    """Function to read the CSV with correct headers
    Parameters
    ----------
    filepath : str
        The file location of the CSV-file
    skier : int
        The skier identity

    Returns
    -------
    df : Dataframe
        A dataframe for the timeseries data
    """
    # Reading CSV and deleting NaN values
    df = pd.read_csv(filepath).dropna()
    # Since NaN values are droped we need to reset the indices
    df = df.reset_index(drop=True)
    # The header information
    df.columns = [
        "Time (min)",
        "Force left (N)",
        "Force right (N)",
        "Time (floor min)",
        "Slope (%)",
        "Speed (km/h)",
        "Gear",
    ]
    # Appending the skier identity for every value
    df["Skier"] = [skier] * len(df.index)
    # Adding timeseries in seconds
    df["Time (sec)"] = df["Time (min)"] * 60
    # Deleting the last values that doesn't have protocol information
    trim = (df["Speed (km/h)"].values == 0) * (df["Time (sec)"].values > 800)
    df = df[~trim]
    return df


def GetStrokes(
    df,
    pole,
    height=0.1,
    width=1,
    distance=64,
    rel_height=0.97,
    cutoff_start=2,
    cutoff_end=2,
):
    """Gets the individual pullings frome the timeseries dataframe
    Parameters
    ----------
    df : dataframe
        Dataframe over timeseries
    pole: str
        Either "Left" or "Right"
    height : float
        Procentage threshold of max force that makes the peak detected
    distance: int
        Minimum distance between individual pullings
    rel_height : float
        Procentage of peak height for detecting where a peak starts. This makes
        the width of the peak
    cutoff_start : float
        Time to cutoff from the begining of the dataframe before starting to
        detect peaks
    cufoff_end : float
        Time to cutoff from the end of the dataframe before starting to detect
        peaks
    Returns
    -------
    df_peaks : dataframe
        Dataframe over individual peaks in a timeseries matter
    """
    # Creating a force string for extration of either left or right
    force_string = "Force " + pole.lower() + " (N)"
    # Creating the proper height
    height = height * np.max(df[force_string])
    # Updating the cutoff at the end
    cutoff_end = df["Time (sec)"].values[-1] - cutoff_end
    # Getting the peaks from force data
    peaks, _ = find_peaks(
        df[force_string], height=height, width=width, distance=distance
    )
    # Getting the start and stop possition of groundcontact, with interpolation
    _, _, left_ips, right_ips = peak_widths(
        df[force_string], peaks, rel_height=rel_height
    )
    # Rounding to get rid of interpolation
    left_ips = np.floor(left_ips)
    right_ips = np.ceil(right_ips)

    # Getting the time values from index values
    ground_start = df["Time (sec)"][left_ips].values
    ground_stop = df["Time (sec)"][right_ips].values
    # Triming for cutoff
    index_trim = np.logical_and(
        ground_start >= cutoff_start, ground_start <= cutoff_end
    )
    ground_start = ground_start[index_trim]

    # Wanting to get a ground start as first value and a ground_stop as last
    # value.
    index_trim = np.logical_and(
        ground_start[0] < ground_stop, ground_stop < ground_start[-1]
    )
    # Updating with the new trim
    peaks = peaks[index_trim]
    left_ips = left_ips[index_trim]
    right_ips = right_ips[index_trim]
    gear = df.Gear[peaks]
    skier = df.Skier[peaks]
    ground_stop = ground_stop[index_trim]
    # Generating the axial impulse
    force_area = [0] * len(peaks)
    for i in range(len(peaks)):
        start = int(left_ips[i])
        stop = int(right_ips[i])
        force_area[i] = np.trapz(
            df[force_string][start:stop].values,
            df["Time (sec)"][start:stop].values,
        )
    # Saving the data to a dataframe
    data = {
        "Gear": gear.values,
        "Peak time": df["Time (sec)"][peaks],
        "Peak height": df[force_string][peaks],
        "Stroke and ground contact start": ground_start[:-1],
        "Height start": [0] * len(ground_start[:-1]),
        "Ground contact stop": ground_stop,
        "Height stop": [0] * len(ground_stop),
        "Stroke stop": ground_start[1:],
        "Skier": skier,
        "Impulse axial": force_area,
        "Pole": pole,
    }
    df_peaks = pd.DataFrame(data)
    # Deleate last value, since algorithm can't pick up where skier ends
    df_peaks.drop(df_peaks.tail(1).index, inplace=True)
    return df_peaks


def GetInfo(df_peaks):
    """ Generating data over individual pulling with a non timeseries matter
    Parameters
    ----------
    df_peaks : dataframe
        Dataframe with data for individual pulling in a timeseries matter

    Returns
    -------
    df_info : dataframe
        Dataframe over individual pulling in a non timeseries matter
    """
    # Time between pushing the pole to the ground
    stroke_time = (
        df_peaks["Stroke stop"].values
        - df_peaks["Stroke and ground contact start"].values
    )
    # Time between pole force to lifting it up
    ground_contact_time = (
        df_peaks["Ground contact stop"].values
        - df_peaks["Stroke and ground contact start"].values
    )
    # Time for recovery
    air_time = (
        df_peaks["Stroke stop"].values - df_peaks["Ground contact stop"].values
    )
    # Saving to dataframe
    data = {
        "Gear": df_peaks["Gear"].values,
        "Stroke time": stroke_time,
        "Pull time": ground_contact_time,
        "Recovery time": air_time,
        "Frequency": 1 / stroke_time,
        "Impulse axial": df_peaks["Impulse axial"].values,
        "Skier": df_peaks["Skier"].values,
        "Peak time": df_peaks["Peak time"].values,
        "Pole": df_peaks["Pole"].values,
    }
    df_info = pd.DataFrame(data)
    return df_info


def add_left_and_right_diff(
    df_peaks_left, df_peaks_right, df_info_left, df_info_right
):
    """ Algoritm for getting procentage between left and right pullings of the
    whole stroke and height difference
    Parameters
    ----------
    df_peaks_left : dataframe
        Timeseries data for individual left pullings
    df_peaks_right : dataframe
        Timeseries data for individual right pullings
    df_info_left : dataframe
        Non timeseries data for individual left pullings
    df_info_right : dataframe
        Non timeseries data for individual right pullings
    Returns
    -------
    df_info_left : dataframe
        Non timeseries data for individual left pullings
    df_info_right : dataframe
        Non timeseries data for individual right pullings
    """
    # Creating a Nearest neighbor mapping for the left peaks
    nbrs_left = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(
        np.transpose([df_peaks_left["Peak time"].values])
    )
    # Creating a Nearest neighbor mapping for the right peaks
    nbrs_right = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(
        np.transpose([df_peaks_right["Peak time"].values])
    )
    # Getting the times from the right peaks to the left peaks
    times_right, indices_right = nbrs_left.kneighbors(
        np.transpose([df_peaks_right["Peak time"].values])
    )
    # Getting the times from the left peaks to the right peaks
    times_left, indices_left = nbrs_right.kneighbors(
        np.transpose([df_peaks_left["Peak time"].values])
    )
    # Save to dataframe
    df_info_left["Time to other pole"] = np.transpose(times_left)[
        0
    ]  # / df_info_left["Stroke time"].values
    df_info_right["Time to other pole"] = np.transpose(times_right)[
        0
    ]  # / df_info_right["Stroke time"].values

    df_info_left["Other pole time"] = df_info_right.loc[
        indices_left.flatten(), "Peak time"
    ].values
    df_info_right["Other pole time"] = df_info_left.loc[
        indices_right.flatten(), "Peak time"
    ].values

    # Saving the height of the peak to a variable from dataframe
    height_left = df_peaks_left["Peak height"].reset_index(drop=True)
    height_right = df_peaks_right["Peak height"].reset_index(drop=True)
    # Getting the which height this corresponds to for the opposite pole
    height_coresponding_left = height_right.iloc[indices_left.flatten()]
    height_coresponding_right = height_left.iloc[indices_right.flatten()]
    # Calculating the height diff as a procentage of the min value divided by
    # max value
    df_info_left["Peak height diff"] = np.abs(
        height_coresponding_left.values - height_left.values
    ) / np.max([height_coresponding_left.values, height_left.values])
    df_info_right["Peak height diff"] = np.abs(
        height_coresponding_right.values - height_right.values
    ) / np.max([height_coresponding_right.values, height_right.values])

    return df_info_left, df_info_right


def delete_outlier(
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
    """ Function for deleting outliers
    Parameters
    ----------
    trim_features : str list
        The features that we want to delete outliers for
    number_of_oly_up : int
        Number of features that we wishes to only delete outliers from the
        largest values
    quantile : float
        What quantile we wishes to delete, both as quantile but also 1-quantile
    Returns
    -------
    df : dataframe
        Dataframe with deleted outliers
    """
    # Values for the low (down) and high (up) values for the different features
    up = [0] * len(trim_features)
    down = [0] * len(trim_features)
    for i in range(len(trim_features)):
        up[i] = df[trim_features[i]].quantile(1 - quantile)
        down[i] = df[trim_features[i]].quantile(quantile)
    # Deleting the outliers
    for i in range(len(trim_features)):
        if i != len(trim_features) - number_of_only_up:
            df = df[df[trim_features[i]] > down[i]]
        df = df[df[trim_features[i]] < up[i]]
    return df
