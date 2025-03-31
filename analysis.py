#!/usr/bin/env python
# coding: utf-8
import gc
import logging
import matplotlib.pyplot as plt

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

permitted_cols = ['seconds_elapsed', 'accelerometer_z', 'accelerometer_y',
       'accelerometer_x', 'gravity_z', 'gravity_y', 'gravity_x', 'gyroscope_z',
       'gyroscope_y', 'gyroscope_x', 'orientation_qz', 'orientation_qy',
       'orientation_qx', 'orientation_qw', 'orientation_roll',
       'orientation_pitch', 'orientation_yaw', 'magnetometer_z',
       'magnetometer_y', 'magnetometer_x', 'compass_magneticBearing',
       'barometer_relativeAltitude', 'barometer_pressure',
       'location_bearing', 'location_altitude', 'location_longitude', 
       'location_latitude', 'microphone_dBFS', 'light_lux', 'batteryTemp_temperature']

col_synonyms = ['seconds_elapsed', 'Acc.z', 'Acc.y',
       'Acc.x', 'Grav.z', 'Grav.y', 'Grav.x', 'Gyr.z',
       'Gyr.y', 'Gyr.x', 'Ori.qz', 'Ori.qy',
       'Ori.qx', 'Ori.qw', 'Ori.roll',
       'Ori.pitch', 'Ori.yaw', 'Mag.z',
       'Mag.y', 'Mag.x', 'Comp.bear',
       'Bar.alt', 'Bar.press', # Bar = Barometer. Alt = Altitude
       'Loc.bear', 'Loc.alt', 'Loc.long', 
       'Loc.lat', 'Sound', 'Light', 'Temp']

def print_nice_heading(text: str):
    # A heading: centered, bold magenta, with a rule above and below
    heading_text = Text(text, style="bold magenta")
    console.rule(characters="=")
    console.print(heading_text, justify="center")
    console.rule(characters="=")


def print_nice_subheading(text: str):
    # A subheading: left-aligned, bold cyan, and a rule afterwards
    subheading_text = Text(text, style="bold cyan")
    console.rule(characters="-")
    console.print(subheading_text)
    console.rule(characters="-")


def load_data_file(file_path, root_dir=None):
    df = pd.read_csv(file_path, usecols=permitted_cols).dropna()
    return df.rename(columns=dict(zip(permitted_cols, col_synonyms)))


def plot_correlation_matrix(df, prefix=None, save=True):
    sns.set_theme(style="whitegrid")
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df.corr(), cmap=cmap)
    plt.show()


def shannon_entropy(series, base=2):    
    # Calculate the frequency of each unique value.
    counts = series.value_counts()
    
    # Convert counts to probabilities.
    probabilities = counts / counts.sum()
    
    # Calculate the entropy using the formula:
    # H(X) = - sum(p_i * log(p_i))
    # The logarithm base can be adjusted; here, we use natural log and convert by dividing by log(base).
    return -np.sum(probabilities * np.log(probabilities)) / np.log(base)


def expected_guessing_entropy(series):    
    # Compute the frequency of each unique value.
    # .value_counts() returns counts sorted in descending order by default.
    counts = series.value_counts()
    
    # Convert counts to probabilities.
    probabilities = counts / counts.sum()
    
    # Ensure probabilities are sorted in descending order.
    probabilities = probabilities.sort_values(ascending=False)
    
    # Create an array of ranks starting from 1 to the number of unique outcomes.
    ranks = np.arange(1, len(probabilities) + 1)
    
    # Return expected guessing entropy as the weighted sum of ranks.
    return np.sum(ranks * probabilities.values)


def marginal_guesswork(series, alpha=0.95):
    # Calculate frequency counts (sorted descending by default).
    counts = series.value_counts()
    
    # Convert counts to probabilities.
    probabilities = counts / counts.sum()
    
    # Compute cumulative probabilities.
    cum_probs = probabilities.cumsum().values
    
    # When alpha is 1 (or nearly 1), return the total number of unique outcomes.
    if np.isclose(alpha, 1.0):
        return len(cum_probs)
    
    # Find the smallest index k where the cumulative probability >= alpha.
    # np.argmax returns the first occurrence where the condition is True.
    k = np.argmax(cum_probs >= alpha) + 1  # +1 because guesses are 1-indexed.
    return k


def massey_bound(series):
    return 0.25 * ((2**shannon_entropy(series))-1)


def display_entropies(series):
    display(f"H(X)    = {shannon_entropy(series)}")
    display(f"E[G(X)] ≥ {massey_bound(series)}")
    display(f"E[G(X)] = {expected_guessing_entropy(series)}")


def simulate_average_guesses(series, n=1000):    
    # Compute frequency counts; value_counts() returns counts sorted in descending order.
    counts = series.value_counts()
    
    # The guess order is fixed:
    guess_order = counts.index.to_numpy()  # Unique outcomes in sorted order (most frequent first)
    
    # Compute the probabilities corresponding to the guess order.
    probabilities = counts.to_numpy() / counts.to_numpy().sum()
    
    # Instead of looping, sample indices (representing the rank) directly in a vectorized manner.
    # Since guess_order is in descending order, if an outcome is at index i, the number of guesses is i+1.
    sampled_indices = np.random.choice(len(guess_order), size=n, p=probabilities)
    # Compute the average number of guesses (adding 1 because ranks are 0-indexed).
    avg_guesses = np.mean(sampled_indices + 1)
    return sampled_indices, avg_guesses


def min_entropy(series):    
    # Calculate the frequency of each unique value.
    counts = series.value_counts()
    
    # Convert counts to probabilities.
    probabilities = counts / counts.sum()
    
    # Get the maximum probability
    max_prob = probabilities.max()
    # Compute and return the min-entropy using base 2 logarithm
    return -np.log2(max_prob)
    

def sliding_window_joint_guesswork_target(reference_guess_order, target_df, sensor_cols, window_size, step=1, default_rank=None):
    results = []
    if default_rank is None:
        default_rank = max(reference_guess_order.values()) + 1

    # Slide the window over the target DataFrame.
    for start in range(0, len(target_df) - window_size + 1, step):
        window = target_df.iloc[start:start + window_size]
        #guess



def display_entropy_results(df):
    results = []
    # Iterate over the columns of interest and compute the values.
    for col in df.columns[1:]:
        series = df[col]
        entropy = shannon_entropy(series)
        massey = massey_bound(series)
        exp_guess = expected_guessing_entropy(series)
        min_entropy_val = min_entropy(series)
        alpha_95_guesswork = marginal_guesswork(series)
        
        results.append({
            'Sensor': col,
            'H(X)': entropy,
            'H∞(X)': min_entropy_val,
            'E[G(X)]': exp_guess,
            'α (95%)': alpha_95_guesswork
        })
    # Create a DataFrame from the results and display it.
    results_df = pd.DataFrame(results)
    display(results_df.round(3))


def single_sensor_analysis():
    data_dirs = ["garden"] # "auto", "home", "office"
    path = "data/garden/GardenPixel-2025-03-23_17-03-12-8132b4b89fb54aa7a9762a4a82d6863d.csv"
    print(f"> Entropy values for {path}")

    df = load_data_file(path)
    
    # Correlation matrix
    #plot_correlation_matrix(df.drop("seconds_elapsed", axis=1), prefix=data_dirs[0], save=True)

    display_entropy_results(df)

    
    """
    print("> Quantising dataframe")
    uci_har_joint_entropy_df = entropies_of_sensor_combinations(
        quantise_df(joint_sensors_df))
    uci_har_joint_entropy_df.to_csv("uci-har_results.csv", index=False)

    display_top_and_bottom_n(uci_har_joint_entropy_df, name="UCI-HAR", colsort=["H_inf"])
    plot_distribution_entropies(uci_har_joint_entropy_df["H_inf"].values.flatten(),
                                name="uci-har", graph_dir=GRAPH_OUTPUT)
   """
    

if __name__ == "__main__":
    print_nice_heading("Practical Guessing Attacks by Carlton Shepherd (2025, https://cs.gl)")
    print_nice_subheading("Single Sensor Analysis")
    single_sensor_analysis()