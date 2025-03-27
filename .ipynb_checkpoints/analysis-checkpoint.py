#!/usr/bin/env python
# coding: utf-8
import gc
import logging

from utils import *
from pyfiglet import Figlet
from tqdm import tqdm
from prettytable import PrettyTable
from IPython.display import display, Markdown
from rich.console import Console
from rich.text import Text

DATASET_PATH="./data"
GRAPH_OUTPUT="./output"

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

def data_analysis():
    data_dirs = ["garden"] # "car", "home", "office"
    
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
    

if __name__ == "__main__":
    print_nice_heading("Practical Guessing Attacks by Carlton Shepherd (2025, https://cs.gl)")
    print_nice_subheading("UCI-HAR Dataset")
    uci_har_analysis()
    print_nice_subheading("Relay Dataset")
    relay_analysis()
    print_nice_subheading("Finished")
