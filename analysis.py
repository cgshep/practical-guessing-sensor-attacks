#!/usr/bin/env python
# coding: utf-8

# # Entropy Analysis
# 
# This notebook computes entropy values for the sensor data in the UCI-HAR, SHL datasets.
# 
# UCI-HAR
# The dataset contains accelerometer and gyroscope readings for 30 user subjects.
#
import gc
import logging

from utils import *
from pyfiglet import Figlet
from tqdm import tqdm
from prettytable import PrettyTable
from IPython.display import display, Markdown
from rich.console import Console
from rich.text import Text

DATASET_DIR="./datasets"
UCIHAR_PATH=f"{DATASET_DIR}/ucihar"
INERTIAL_SIGNALS=f"{UCIHAR_PATH}/train/Inertial Signals"
GRAPH_OUTPUT="./output"
SAMPLE_FACTOR = 0.001

logger = logging.getLogger()
logger.setLevel(logging.INFO)

console = Console()

def print_nice_heading(text: str):
    # A heading: centered, bold magenta, with a rule above and below
    heading_text = Text(text, style="bold magenta")
    console.rule(characters="=")       # A line with '='
    console.print(heading_text, justify="center")
    console.rule(characters="=")

def print_nice_subheading(text: str):
    # A subheading: left-aligned, bold cyan, and a rule afterwards
    subheading_text = Text(text, style="bold cyan")
    console.rule(characters="-")
    console.print(subheading_text)
    console.rule(characters="-")

def uci_har_analysis():
    subject_ids = pd.read_table(f"{INERTIAL_SIGNALS}/../subject_train.txt",
                                sep=r"\s+", header=None)

    feature_names = pd.read_table(f"{UCIHAR_PATH}/features.txt", sep=r"\s+",
                                  header=None, names=('ID','Feature'))

    axes = ['x', 'y', 'z']
    acc = {axis: load_uci_har_signal(f"total_acc_{axis}_train.txt",
                                     root_dir=INERTIAL_SIGNALS) for axis in axes}
    gyro = {axis: load_uci_har_signal(f"body_gyro_{axis}_train.txt",
                                      root_dir=INERTIAL_SIGNALS) for axis in axes}


    # We've loaded the data into the `acc` and `gyro` dicts, and we have a list of
    # subject IDs for each row in `subject_ids` if needed. Let's produce some histograms.

    matplotlib.rcParams.update({'font.size': 16})
    sensors=["acc", "gyro"]

    for s in sensors:
        for i, ax in enumerate(axes):
            if s == "acc":
                s_ax = acc[ax]
                x_label = "Acceleration ($ms^{-2}$)"
            elif s == "gyro":
                s_ax = gyro[ax]
                x_label = "Angular Velocity ($rads^{-1}$)"
            plot_and_save_graph(s_ax, x_label,
                                graph_dir=GRAPH_OUTPUT,
                                file_name=f"{s}_{ax}",
                                i=3,
                                save=True)



    t = PrettyTable(["Modality", "H_0", "H_1", "H_2", "H_inf"])
    t.align = "l"
    for s in sensors:
        if s == "acc":
            s_ax = acc
        elif s == "gyro":
            s_ax = gyro
        x_flat = s_ax["x"].values.flatten()
        y_flat = s_ax["y"].values.flatten()
        z_flat = s_ax["z"].values.flatten()

        # Ensure that x, y, and z have the same number of samples
        assert len(x_flat) == len(y_flat) == len(z_flat)
        print_entropies(entropy_with_adaptive_binning(x_flat), f"{s}.x", t)
        print_entropies(entropy_with_adaptive_binning(y_flat), f"{s}.y", t)
        print_entropies(entropy_with_adaptive_binning(z_flat), f"{s}.z", t)
        print_entropies(entropy_with_adaptive_binning(mag(x_flat, y_flat, z_flat)),
                        f"{s}.mag", t)

    print("> Single sensor entropies:")
    print(t)
    with open(f"output/uci-har_single_sensor_entropies.txt", "w") as f:
        f.write(str(t))

    # Joint Entropy
    #
    # See "features.txt" and "features_info.txt" for more info about the
    # feature indices in UCI-HAR.
    #
    joint_axes = [0, 1, 2, 120, 121, 122]
    joint_col_names = ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]
    joint_sensors_df = load_uci_har_signal("../X_train.txt",
                                           root_dir=INERTIAL_SIGNALS,
                                           usecols=joint_axes,
                                           colnames=joint_col_names)
    plot_correlation_matrix(joint_sensors_df, prefix="uci-har_", save=True)

    print("> Quantising dataframe")
    uci_har_joint_entropy_df = entropies_of_sensor_combinations(
        quantise_df(joint_sensors_df))
    uci_har_joint_entropy_df.to_csv("uci-har_results.csv", index=False)
    del joint_sensors_df
    gc.collect()

    display_top_and_bottom_n(uci_har_joint_entropy_df, name="UCI-HAR", colsort=["H_inf"])
    plot_distribution_entropies(uci_har_joint_entropy_df["H_inf"].values.flatten(),
                                name="uci-har", graph_dir=GRAPH_OUTPUT)


def relay_analysis():
    """
    Analyses the entropy of sensor data for detecting relay attacks on
    NFC contactless transactions by Gurulian et al.

    Data available from https://github.com/AmbientSensorsEvaluation/Ambient-Sensors-Relay-Attack-Evaluation/tree/master
    """
    RELAY_DATA_DIR = f"{DATASET_DIR}/relay"

    # Creates an array of tuples of the form (sensor name, data CSV)
    sensor_filepaths = [(re.findall("^.*/([a-zA-Z]+).csv$", f_path)[0], f_path) 
                        for f_path in glob.glob(f"{RELAY_DATA_DIR}/*.csv")]

    sensor_units = {
        "MagneticField" : "Magnetic Flux Density ($\mu T$)",
        "Gyroscope" : "Angular Velocity ($rads^{-1}$)",
        "Accelerometer" : "Acceleration ($ms^{-2}$)",
        "Light" : "Illuminance (lux)",
        "RotationVector" : "Rotation Vector",
        "Gravity" : "Acceleration ($ms^{-2}$)",
        "LinearAcceleration" :  "Acceleration ($ms^{-2}$)",
        "Humidity" : "% RH",
        "Temperature" : "Celcius",
        "Pressure" : "Bar"
    }

    t = PrettyTable(["Modality", "H_0", "H_1", "H_2", "H_inf"])
    t.align = "l"
    combined_dict = {}
    for sensor_name, file_path in tqdm(sensor_filepaths, desc="Loading Relay sensor data"):
        df = pd.read_csv(file_path)
        combined_dict[sensor_name] = df.drop(["genuine_transaction"], axis=1)

    # Single sensor analysis
    combined_df = None
    for sensor_name, v in tqdm(combined_dict.items(),
                               desc="Computing single sensor entropies (Relay)"):
        v["sensor_name"] = sensor_name
    
        if combined_df is None:
            # Initialize combined_df with the first DataFrame
            combined_df = v
        else:
            # Concatenate subsequent DataFrames
            combined_df = pd.concat([combined_df, v], ignore_index=True)
        
        df = v.drop(["shared_id", "sensor_name"], axis=1)
        vals = df.values.flatten()
        print_entropies(entropy_with_adaptive_binning(vals), sensor_name, t)

    #plot_and_save_graph(df, file_name=sensor_name, x_label=sensor_units[sensor_name], i=3)
    print(t)
    with open(f"output/relay_single_sensor_entropies.txt", "w") as f:
        f.write(str(t))

    print("> Data wrangling...")
    filtered_df = combined_df[combined_df["shared_id"].str.contains("ALL_SEN_", na=False)].copy()

    # Some data wrangling to make our "shared_id"s unique for the purpose of this analysis
    filtered_df = pd.melt(filtered_df, id_vars=["shared_id", "sensor_name"])

    # The data set is ambiguous such that there are multiple "ALL_SEN_*" shared IDs,
    # even though the data was collected in different set ups. Thus, assume that the 
    # order matters and rename the shared_id column accordingly.
    filtered_df["suffix"] = filtered_df.groupby(["sensor_name", "shared_id"]).cumcount() % 4
    filtered_df["suffix_1"] = filtered_df.groupby(["shared_id"]).cumcount() % 2
    filtered_df["variable_numeric"] = pd.factorize(filtered_df["variable"])[0]
    filtered_df["shared_id"] = filtered_df["shared_id"] + '_' + filtered_df["suffix"].astype(str) + \
        '_' + filtered_df["suffix_1"].astype(str) + "_" + filtered_df["variable_numeric"].astype(str)
    filtered_df.drop(["suffix", "suffix_1", "variable_numeric", "variable"], inplace=True, axis=1)

    # Pandas doesn't handle pivot tables very nicely; we need to
    # clean up the structure of our dataframe columns post-pivot
    filtered_df = filtered_df.pivot(index="shared_id", columns="sensor_name")
    filtered_df.columns = [col[1] for col in filtered_df.columns]  # Remove the first level from MultiIndex
    filtered_df.reset_index(inplace=True)

    # Fix the sorting issue
    filtered_df['sort_key'] = filtered_df['shared_id'].str.extract(r'_(\d+)$')[0].astype(int)
    filtered_df['shared_id_base'] = filtered_df['shared_id'].str.extract(r'^(.*)_\d+$')[0]
    relay_df = filtered_df.sort_values(by=['shared_id_base', 'sort_key']).drop(columns=['shared_id_base', 'sort_key', 'shared_id'])
    del filtered_df

    sensor_abbreviations = {
        "Accelerometer" : "Acc.",
        "Gravity" : "Grav.",
        "Gyroscope" : "Gyro.",
        "Light" : "Light",
        "LinearAcceleration" : "Lin. Acc.",
        "MagneticField" : "Mag. Field",
        "RotationVector" : "Rot. Vec."
    }

    # Replace column names with abbreviations.
    relay_df.rename(columns=sensor_abbreviations, inplace=True)
    plot_correlation_matrix(relay_df, prefix="relay_", save=True)
    print("> Done!")
    print("> Quantising dataframe")
    relay_joint_df = entropies_of_sensor_combinations(quantise_df(relay_df))
    relay_joint_df.to_csv("relay_results.csv", index=False)

    display_top_and_bottom_n(relay_joint_df, name="Relay", colsort=["H_inf"])
    plot_distribution_entropies(relay_joint_df["H_inf"].values.flatten(),
                                name="relay", graph_dir=GRAPH_OUTPUT)


def shl_analysis(parallelise=True):
    """
    This is an analysis of the Sussex-Huawei Preview dataset comprising sensor
    data from $n=3$ users. See the docs.

    Note: The current dataset is far too high a dimensionality (d=22) even
    for our estimation approach. To reduce this, we use sampling-based
    approach; it makes it computationally feasible.
    """
    SHL_DIR="/media/carlton/Data/shl"
    users = ["user1", "user2", "user3"]
    master_df = None

    for file_path in tqdm(glob.glob(f"{SHL_DIR}/Hand_Motion_*.txt"),
                          desc="Loading SHL Data"):
        df = pd.read_table(file_path,
                           sep=r"\s+",
                           header=None,
                           on_bad_lines="warn",
                           memory_map=True,
                           dtype=np.float32)
        if master_df is None:
            master_df = df
        else:
            master_df = pd.concat([master_df, df])

    master_df.dropna(axis=0, how="any", inplace=True)

    # Cleanup time! Do a bit of column renaming to make things easier
    column_rename_dict = {0: "timestamp",
                          1 : "Acc.x",
                          2 : "Acc.y",
                          3 : "Acc.z",
                          4 : "Gyro.x",
                          5 : "Gyro.y",
                          6 : "Gyro.z",
                          7 : "Mag.x",
                          8 : "Mag.y",
                          9 : "Mag.z",
                          10 : "Ori.w",
                          11 : "Ori.x",
                          12 : "Ori.y",
                          13 : "Ori.z",
                          14 : "Grav.x",
                          15 : "Grav.y",
                          16 : "Grav.z",
                          17 : "LinAcc.x",
                          18 : "LinAcc.y",
                          19 : "LinAcc.z",
                          20 : "Pressure",
                          21 : "Altitude",
                          22 : "Temp." }

    master_df.rename(columns=column_rename_dict, inplace=True)
    master_df.drop("timestamp", axis=1, inplace=True)
    plot_correlation_matrix(master_df, prefix="shl_", save=True)

    triaxial_sensors = ["Acc.", "Gyro.", "Mag.", "Ori.", "Grav.", "LinAcc."]
    t = PrettyTable(["Modality", "H_0", "H_1", "H_2", "H_inf"])
    t.align = "l"

    for sensor in triaxial_sensors:
        # Strip the trailing '.' to match how columns are named in master_df
        sensor_prefix = sensor.rstrip('.')

        # Build column names for x, y, z
        axis_columns = [f"{sensor_prefix}.{axis}" for axis in ["x", "y", "z"]]

        # Create a new column for the magnitude
        mag_column = f"{sensor_prefix}.mag"
        master_df[mag_column] = np.sqrt(master_df[axis_columns].pow(2).sum(axis=1))

    for col in tqdm(master_df.columns, desc="Computing single sensor entropies (SHL)"):
        vals = master_df[col].values.flatten()
        print_entropies(entropy_with_adaptive_binning(vals), f"{col}", t)

    print(t)
    with open(f"output/shl_single_sensor_entropies.txt", "w") as f:
        f.write(str(t))
    
    master_df.drop(["Acc.mag", "Gyro.mag", "Mag.mag",
                    "Ori.mag", "Grav.mag", "LinAcc.mag"], axis=1, inplace=True)

    print(f"> Quantising dataframe with shape {master_df.shape} and sampling {SAMPLE_FACTOR}")
    mask = np.random.rand(len(master_df)) < SAMPLE_FACTOR
    quantised_df = quantise_df(master_df[mask])
    print(f"> Computing joint entropies of dataset with shape {quantised_df.shape}")

    if parallelise:
        logger.info("Using parallel joint entropy calculation")
        shl_joint_entropy_all = parallel_combination_entropies(quantised_df,
                                                               skip_range=(3,19))
    else:
        logger.info("Using single-threaded joint entropy calculation")
        shl_joint_entropy_all = entropies_of_sensor_combinations(quantised_df,
                                                                 skip_range=(3,19))

    display_top_and_bottom_n(shl_joint_entropy_all, name="SHL", colsort=["H_inf"])
    plot_distribution_entropies(shl_joint_entropy_all["H_inf"].values.flatten(),
                                name="SHL", graph_dir=GRAPH_OUTPUT)
    shl_joint_entropy_all.to_csv("shl_results.csv", index=False)


def perilzis_analysis():
    """
    Compute single and multi-modal entropies for the PerilZIS dataset by Formichev et al.

    Data available at:
    
    """
    PERIL_DIR=f"{DATASET_DIR}/perilzis/merged_sensors"

    peril_sensor_aliases = {
    "mag": "MagneticField",
        "gyr": "Gyroscope",
        "acc": "Accelerometer",
        "hum" : "Humidity",
        "lux" : "Light",
        "bar" : "Pressure",
        "tmp" : "Temperature"
    }

    triaxial_sensors = ["mag", "gyr", "acc"]
    triaxial_axes = ["x", "y", "z"]
    t = PrettyTable(["Modality", "H_0", "H_1", "H_2", "H_inf"])
    t.align = "l"

    def peril_single_sensor_analysis(df, sensor_str):
        # Drop the last column; it's a timestamp
        df1 = df[df.columns[:-1]].dropna()

        if sensor_str in triaxial_sensors:
            # For triaxial sensors, do each axis individually
            for i, axis in enumerate(triaxial_axes):
                vals = df1.iloc[:, i].values.flatten()
                print_entropies(entropy_with_adaptive_binning(vals),
                                f"{peril_sensor_aliases[sensor_str]}.{axis}", t)
            #
            # Derive the magnitude synthetic "sensor"
            #
            df1["mag"] = np.sqrt(df[[0, 1, 2]].pow(2).sum(axis=1))
            vals = df1["mag"].values.flatten()
            print_entropies(entropy_with_adaptive_binning(vals),
                            f"{peril_sensor_aliases[sensor_str]}.mag", t)
        else:
            vals = df1.values.flatten()
            print_entropies(entropy_with_adaptive_binning(vals),
                            f"{peril_sensor_aliases[sensor_str]}", t)
    #plot_and_save_graph(df1, file_name=sensor_str, x_label=sensor_units[peril_sensor_aliases[sensor_str]], i=3, save=False)
    
    peril_data = {}

    for file_path in tqdm(glob.glob(f"{PERIL_DIR}/all_*.txt"),
                          desc="Computing Single Sensor Entropies (PerilZIS)"):
        sensor_str = re.findall(r"^.*\/all_([a-zA-Z]+).txt$", file_path)[0]
        df = pd.read_table(file_path, sep=r"\s+", header=None, 
                           on_bad_lines="warn",
                           memory_map=True)
        # Keep track of the read data for when we do our multi-modal analysis
        peril_data[sensor_str] = df
        peril_single_sensor_analysis(df, sensor_str)
    print(t)
    with open(f"output/perilzis_single_sensor_entropies.txt", "w") as f:
        f.write(str(t))

    # Joint Entropy
    master_df = None
    for sensor_name, df in tqdm(peril_data.items(),
                                total=len(peril_data.items()),
                                desc="Cleaning dataframe"):
        df = df.add_prefix(f"{sensor_name}_")
        df["rounded_timestamp"] = pd.to_datetime(df.iloc[:,-1],
                                                 format="ISO8601").dt.round('250ms')
        df.drop(df.columns[-2], axis=1, inplace=True)
        df.drop_duplicates(subset=["rounded_timestamp"], inplace=True)

        if master_df is None:
            master_df = df
        else:
            master_df = pd.merge(master_df, df, on="rounded_timestamp", how="inner")
    del peril_data
    gc.collect()

    master_df.drop("rounded_timestamp",axis=1,inplace=True)

    # Cleanup time! Do a bit of column renaming to make things easier
    column_rename_dict = { "hum_0": "Humidity",
                           "acc_0" : "Acc.x",
                           "acc_1" : "Acc.y",
                           "acc_2" : "Acc.z",
                           "lux_0" : "Light",
                           "tmp_0" : "Temp.",
                           "bar_0" : "Pressure",
                           "mag_0" : "Mag.x",
                           "mag_1" : "Mag.y",
                           "mag_2" : "Mag.z",
                           "gyr_0" : "Gyro.x",
                           "gyr_1" : "Gyro.y",
                           "gyr_2" : "Gyro.z" }

    master_df.rename(columns=column_rename_dict, inplace=True)
    master_df.fillna(master_df.mean(), inplace=True)
    plot_correlation_matrix(master_df, prefix="perilzis_", save=True)

    print("> Quantising dataframe")
    perilzis_joint_entropy_df = parallel_combination_entropies(quantise_df(master_df))
    del master_df
    gc.collect()

    display_top_and_bottom_n(perilzis_joint_entropy_df, name="PerilZIS", colsort=["H_inf"])
    plot_distribution_entropies(perilzis_joint_entropy_df["H_inf"].values.flatten(),
                                name="perilzis", graph_dir=GRAPH_OUTPUT)
    perilzis_joint_entropy_df.to_csv("perilzis_results.csv", index=False)
    

if __name__ == "__main__":
    print_nice_heading("Entropy Analyser by Carlton Shepherd (2025, https://cs.gl)")
    # You'll want to get these datasets first. They're VERY big.
    # In general, you can add your own dataset by having sensor modalities as
    # columns with the rows containing data sampled at the same time.
    # The UCI-HAR and Relay datasets are small, so they're given in the git repo.
    #print_nice_subheading("SHL Dataset")
    #shl_analysis()
    #print_nice_subheading("PerilZIS Dataset")
    #perilzis_analysis()
    print_nice_subheading("UCI-HAR Dataset")
    uci_har_analysis()
    print_nice_subheading("Relay Dataset")
    relay_analysis()
    print_nice_subheading("Finished")
