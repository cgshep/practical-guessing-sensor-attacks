import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import glob
import re
import logging
import itertools
import seaborn as sns
import networkx as nx
import math

from functools import partial
from multiprocessing import Pool

from tqdm import tqdm
from p_tqdm import p_map
from functools import lru_cache
from pyitlib import discrete_random_variable as drv
from prettytable import PrettyTable
from pgmpy.estimators import TreeSearch
from pgmpy.models import BayesianNetwork
from IPython.display import display, Markdown
from more_itertools import chunked
#from more_itertools import powerset

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BINS=512
ADAPTIVE_BINNING_MAX_BINS = 3072


def load_uci_har_signal(file_path,
                        root_dir, 
                        include_subject_ids=False, 
                        usecols=None, 
                        colnames=None):
    return pd.read_table(f"{root_dir}/{file_path}", 
                         sep=r"\s+", 
                         header=None, 
                         usecols=usecols, 
                         names=colnames,
                         dtype=np.float32)

    
def plot_and_save_graph(data, x_label, file_name, graph_dir, i=0, save=False, show=False):
    colors=["blue", "red", "green", "black"]
    f = plt.figure()
    flattened_vals = data.values.flatten()
    plt.hist(flattened_vals, bins=BINS, color=colors[i], alpha=0.4, density=True, cumulative=True)  
    plt.xlabel(x_label)
    plt.ylabel("Probability Density")
    plt.grid()
    if show:
        plt.show()
    if save:
        f.savefig(f"{graph_dir}/{file_name}_cdf.pdf",bbox_inches="tight")
    plt.close(f)



def renyi_entropy(probabilities, alpha=1.0, base=2.0):
    """
    Compute the Rényi entropy of order alpha for a given probability distribution,
    using NumPy arrays for efficiency.
    
    Parameters
    ----------
    probabilities : array-like
        A NumPy array (or array-like) of probabilities that sum to 1.
    alpha : float, optional
        The order of the Rényi entropy. Must be > 0.
        Default is 1.0, in which case it coincides with the Shannon entropy.
    base : float, optional
        The logarithm base used for entropy measurement. 
        Common choices are 2 (bits) or np.e (nats).
        Default is 2.

    Returns
    -------
    float
        The Rényi entropy of the distribution for the given alpha.
    """
    # Convert to a NumPy array (float type) in case it's not already
    probabilities = np.asarray(probabilities, dtype=float)
    
    # Convert to a NumPy array (float type) in case it's not already
    if not isinstance(probabilities, np.ndarray):
        raise ValueError("Probabilities must be given as a NumPy array!")
    
    # Ensure probabilities sum to 1 (within a tolerance)
    if not np.isclose(probabilities.sum(), 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError("Probabilities must sum to 1.")

    # Handle min-entropy directly if alpha is infinite
    if np.isinf(alpha):
        # H_inf = -log_base(max_i p_i)
        return -np.log(probabilities.max()) / np.log(base)

    # Handle alpha = 0 (Max entropy)
    if np.isclose(alpha, 0.0, atol=1e-12):
        # Count the number of non-zero probabilities
        support_size = np.count_nonzero(probabilities)
        if support_size == 0:
            raise ValueError("Log(0) is undefined!")
        return np.log(support_size) / np.log(base)
    
    # Handle Shannon entropy if alpha = 1
    if np.isclose(alpha, 1.0, rtol=1e-9):
        return -np.sum(probabilities * np.log(probabilities)) / np.log(base)
    
    # General Rényi entropy formula:
    #   H_alpha = (1 / (1 - alpha)) * log_base( sum_i p_i^alpha )
    sum_p_alpha = np.sum(probabilities**alpha)
    return (1.0 / (1.0 - alpha)) * (np.log(sum_p_alpha) / np.log(base))


def entropy_with_adaptive_binning(data,
                                  bins=BINS,
                                  adaptive_binning=True,
                                  max_bins=ADAPTIVE_BINNING_MAX_BINS,
                                  bin_method="auto",
                                  direct_compute=False):
    """
    Calculate the joint entropy of n-dimensional data using binning.

    Parameters:
        data (np.ndarray): 2D array with shape (n_samples, n_dimensions).
        bins (int or sequence of int): Number of bins for each dimension.

    Returns:
        float: Joint entropy in bits.
    """
    if data.ndim == 1:
        # For 1D data, use np.histogram
        if adaptive_binning:
            bins = np.histogram_bin_edges(data, bins=bin_method)
           # if len(bins) > max_bins:
           #     logging.warning(f"Adaptive Binning reduced to max_bins={max_bins}")
           #     bins = max_bins
        
        counts, _ = np.histogram(data, bins)
    elif data.ndim >= 2 and direct_compute:
        # For n-dimensional data, use np.histogramdd
        if adaptive_binning:
            bins = []
            for dim in range(data.shape[1]):
                edges = np.histogram_bin_edges(data[:, dim], bins=bin_method)
                # Limit the number of bins per dimension to max_bins
                if len(edges) - 1 > max_bins:
                    edges = np.histogram_bin_edges(data[:, dim], bins=max_bins)
                    logging.warning(f"Dimension {dim}: Adaptive Binning reduced to max_bins={max_bins}")
                bins.append(edges)
        counts, _ = np.histogramdd(data, bins)
    else:
        raise ValueError("Data must be a 1D or 2D array.")

    # Normalise to get the probability distribution
    total_counts = np.sum(counts)
    if total_counts == 0:
        raise ValueError("The histogram counts sum to zero.")
    prob_dist = counts / total_counts
    
    # Filter out zero probabilities to avoid log(0)
    prob_dist = prob_dist[prob_dist > 0]

    h0 = renyi_entropy(prob_dist, alpha=0)
    h1 = renyi_entropy(prob_dist, alpha=1)
    h2 = renyi_entropy(prob_dist, alpha=2)
    h_inf = renyi_entropy(prob_dist, alpha=np.inf)
    return (h0, h1, h2, h_inf)

def mag(x, y, z):
    return np.sqrt((x*x)+(y*y)+(z*z))

def print_entropies(entropies, 
                    sensor_name, 
                    pretty_table=None, 
                    round_vals=True, 
                    round_decimals=3):
    if round_vals:
        h_0, h_1, h_2, h_inf = (round(h, round_decimals) for h in entropies)
    else:
        h_0, h_1, h_2, h_inf = entropies
    
    if pretty_table:
        pretty_table.add_row([f"{sensor_name}", h_0, h_1, h_2, h_inf])
    else:
        print(f"{sensor_name} - H0: {h_0}, H1: {h_1}, H2: {h_2}, H_inf: {h_inf}")


def plot_correlation_matrix(df, prefix=None, save=False, show=False):
    # Plot  matrix of the df
    corr = df.corr()
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    sns.heatmap(corr, annot=False, cmap='RdBu', fmt=".3f", vmin=-1, vmax=1)
    plt.xticks(rotation=55, ha='right')

    if save:
        plt.savefig(f"{prefix}correlation_matrix.pdf", format="pdf", bbox_inches="tight")
    if show:
        plt.show()

# Quantise
def quantise_df(df):
    for column in df.columns:
        _, bin_edges = np.histogram(df[column], bins="auto")
        if len(bin_edges) > ADAPTIVE_BINNING_MAX_BINS:
            print(f"Capping bin_edges at {ADAPTIVE_BINNING_MAX_BINS} for {column}")
            _, bin_edges = np.histogram(df[column],
                                        bins=ADAPTIVE_BINNING_MAX_BINS)
        df.loc[:, column] = pd.cut(df[column], bins=bin_edges, labels=False, include_lowest=True)
    return df


def compute_max_entropy(model):
    """
    Computes the max-entropy H0(X) of a tree-structured BayesianNetwork.
    
    Parameters:
    - model: BayesianNetwork (pgmpy)
        The Bayesian Network.
    
    Returns:
    - float: The max-entropy in bits.
    """
    # Ensure the model is a forest (collection of trees)
    if not nx.is_forest(model):
        raise ValueError("The Bayesian Network must be a forest (collection of trees).")
    
    # Identify all root nodes (nodes with no parents)
    roots = [node for node in model.nodes() if len(model.get_parents(node)) == 0]
    if not roots:
        raise ValueError("The Bayesian Network must have at least one root node.")
    
    # Initialize a dictionary to store support sizes for each node
    support_sizes = {node: 0 for node in model.nodes()}
    
    # Traverse the network in topological order (parents before children)
    topo_order = list(nx.topological_sort(model))
    
    for node in topo_order:
        cpd = model.get_cpds(node)
        parents = list(model.get_parents(node))
        
        if not parents:
            # Root node: support size is the number of its states
            support_sizes[node] = cpd.cardinality[0]
            #print(f"Node '{node}' is a root with support size {support_sizes[node]}")
        else:
            # Non-root node: support size is the number of states it can take
            # given its parent. Assuming deterministic dependencies or varying
            # support sizes based on parent states is not necessary here,
            # as H0 counts all possible state combinations.plt.xticks(rotation=90)
            # Thus, support size is the number of states regardless of parents.
            support_sizes[node] = cpd.cardinality[0]
            #print(f"Node '{node}' has support size {support_sizes[node]}")
    
    # Compute H0 as the sum of log2 of support sizes
    H0 = sum(math.log2(size) for size in support_sizes.values())
    #print(f"Max-Entropy H0(X): {H0:.4f} bits")
    return H0
    


def compute_collision_entropy_efficient(model):
    """
    Computes the collision (Rényi alpha=2) entropy H2(X) of a tree-structured BayesianNetwork efficiently
    without full enumeration using recursion with memoization.

    Parameters:
    - model: BayesianNetwork (pgmpy)
        The Bayesian Network with fitted CPDs.

    Returns:
    - float: The collision entropy in bits.
    """
    # Ensure the model is a forest (collection of trees)
    if not nx.is_forest(model):
        raise ValueError("The Bayesian Network must be a forest (collection of trees).")

    # Identify all root nodes (nodes with no parents)
    roots = [node for node in model.nodes() if len(model.get_parents(node)) == 0]
    if not roots:
        raise ValueError("The Bayesian Network must have at least one root node.")

    # Create a dictionary for quick access to CPDs
    cpd_dict = {cpd.variable: cpd for cpd in model.get_cpds()}

    @lru_cache(maxsize=None)
    def compute_S(node, state, parent_state):
        """
        Recursively computes the sum of squared probabilities for the subtree rooted at 'node'
        given that 'node' is in 'state' and its parent is in 'parent_state'.

        Parameters:
        - node: The current node in the Bayesian Network.
        - state: The state index of the current node.
        - parent_state: The state index of the parent node. For root nodes, this should be None.

        Returns:
        - float: The sum of squared probabilities for the subtree rooted at 'node'.
        """
        cpd = cpd_dict[node]
        parents = list(model.get_parents(node))

        if not parents:
            # Root node: p(x_i = state)
            p_node = cpd.values.flatten()[state]
        else:
            # Non-root node: p(x_i = state | parent_state)
            parent = parents[0]
            p_node = cpd.values[state, parent_state]

        # Start with the squared probability of the current node
        s = p_node ** 2

        # Iterate over all children of the current node
        for child in model.get_children(node):
            child_cpd = cpd_dict[child]
            sum_child = 0.0
            for child_state in range(child_cpd.cardinality[0]):
                # p(child | node_state)
                p_child_given_i = child_cpd.values[child_state, state]
                # Recursively compute the contribution from the child
                s_child = compute_S(child, child_state, state)
                # Accumulate the sum over child states without additional squaring
                sum_child += s_child
             
            # Multiply the contributions from all children
            s *= sum_child

        return s

    # Initialize the total sum for all trees in the forest
    total_sum = 1.0

    for root in roots:
        root_cpd = cpd_dict[root]
        sum_root = 0.0
        for root_state in range(root_cpd.cardinality[0]):
            # p(root_state)
            p_root = root_cpd.values.flatten()[root_state]
            # Compute the sum of squared probabilities for the subtree rooted at 'root' in 'root_state'
            s = compute_S(root, root_state, None)
            sum_root += s
        # Multiply the contributions from each tree in the forest
        total_sum *= sum_root

    # Handle cases where the total sum is zero or negative
    if total_sum <= 0.0:
        logger.warning("Sum of p(x)^2 is zero or negative, returning -0.0")
        return -0.0

    # Compute Collision Entropy
    H2 = -math.log2(total_sum)
    return H2

def compute_shannon_entropy(model):
    """
    Computes the Shannon entropy H(X1, ..., Xd) of a *tree-structured* BayesianNetwork.
    
    model: BayesianNetwork (pgmpy)
       - Must be a singly connected DAG (each node has at most 1 parent), i.e., Chow–Liu tree.
       - model.fit(...) must have already been called.
    
    Returns: float
       The total Shannon entropy in bits (base 2).
    """
    # Create a dictionary for quick access to CPDs
    cpd_dict = {cpd.variable: cpd for cpd in model.get_cpds()}
    
    # Get a topological ordering of the nodes
    topo_order = list(nx.topological_sort(model))
    
    # Dictionary to store marginal distributions
    distributions = {}
    
    # Initialize total entropy
    total_entropy = 0.0
    
    for node in topo_order:
        parents = list(model.get_parents(node))
        cpd = cpd_dict[node]
        
        if not parents:
            # Root node: Unconditional distribution
            p_x = cpd.values.flatten()
            # Compute entropy: H(X) = -sum p(x) log2 p(x)
            node_entropy = -sum(p * math.log2(p) for p in p_x if p > 0)
            total_entropy += node_entropy
            distributions[node] = p_x
        else:
            # Child node with exactly one parent
            parent = parents[0]
            p_parent = distributions[parent]
            p_child_given_parent = cpd.values
            cond_entropy = 0.0
            p_child = [0.0] * cpd.cardinality[0]
            
            for parent_state_idx, p_p in enumerate(p_parent):
                # Get the conditional distribution for this parent state
                p_x_given_p = p_child_given_parent[:, parent_state_idx]
                # Compute entropy for X given Parent = p
                h_x_given_p = -sum(p * math.log2(p) for p in p_x_given_p if p > 0)
                cond_entropy += p_p * h_x_given_p
                # Update marginal distribution for child
                for child_state_idx, p_x in enumerate(p_x_given_p):
                    p_child[child_state_idx] += p_p * p_x
                #print(f"Conditional entropy H({node}|Parent={parent}) for Parent state {parent_state_idx}: {h_x_given_p:.4f} bits weighted by p_parent={p_p:.4f}")
            
            total_entropy += cond_entropy
            distributions[node] = p_child
            #print(f"Entropy contribution of node '{node}': {cond_entropy:.4f} bits")
    return total_entropy


def compute_min_entropy_efficient(model):
    """
    Computes the min-entropy H_min(X) = -log2( max_x p(x) )
    for a tree-structured BayesianNetwork efficiently without full enumeration.

    Parameters:
    - model: BayesianNetwork (pgmpy)
        The Bayesian Network with fitted CPDs.

    Returns:
    - float: The min-entropy in bits.
    """
    logger = logging.getLogger(__name__)
    
    # Ensure the model is a tree
    if not nx.is_tree(model):
        raise ValueError("The Bayesian Network must be a tree (singly connected DAG).")
    
    # Identify root nodes (should be exactly one in a tree)
    roots = [node for node in model.nodes() if len(model.get_parents(node)) == 0]
    if not roots:
        raise ValueError("The Bayesian Network must have at least one root node.")
        
    if len(roots) > 1:
        raise ValueError("The Bayesian Network must have exactly one root node for a tree.")
    
    root = roots[0]
    
    # Create a dictionary for quick access to CPDs
    cpd_dict = {cpd.variable: cpd for cpd in model.get_cpds()}
    
    @lru_cache(maxsize=None)
    def max_joint_prob(node, parent_state):
        """
        Recursively computes the maximum joint probability for the subtree rooted at 'node'
        given that the parent node is in 'parent_state'.
        
        Parameters:
        - node: The current node in the Bayesian Network.
        - parent_state: The state index of the parent node.
                        For root nodes, this should be None.
        
        Returns:
        - float: The maximum joint probability for the subtree rooted at 'node'.
        """
        cpd = cpd_dict[node]
        parents = list(model.get_parents(node))
        
        if not parents:
            # Root node: p(x_i = state)
            max_p = 0.0
            for state in range(cpd.cardinality[0]):
                p = cpd.values.flatten()[state]
                # For root, no parent, so just p(x)
                # Multiply by children's max probabilities
                children = list(model.get_children(node))
                children_p = 1.0
                for child in children:
                    children_p *= max_joint_prob(child, state)
                total_p = p * children_p
                if total_p > max_p:
                    max_p = total_p
            return max_p
        else:
            # Non-root node: p(x_i = state | parent_state)
            max_p = 0.0
            for state in range(cpd.cardinality[0]):
                p = cpd.values[state, parent_state]
                # Multiply by children's max probabilities
                children = list(model.get_children(node))
                children_p = 1.0
                for child in children:
                    children_p *= max_joint_prob(child, state)
                total_p = p * children_p
                if total_p > max_p:
                    max_p = total_p
            return max_p
    
    # Compute the maximum joint probability starting from the root
    max_p_joint = max_joint_prob(root, None)
    
    if max_p_joint <= 0.0:
        logger.warning("Maximum joint probability is zero or negative, returning -0.0")
        return -0.0
    
    # Compute Min-Entropy
    Hmin = -math.log2(max_p_joint)
    return Hmin




def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    Yields all subsets of 'iterable'.
    """
    s = list(iterable)
    # The chain.from_iterable approach or manual approach:
    # e.g.: ( (s[i] for i in range(len(s)) if (mask & (1 << i))) for mask in range(1 << len(s)) )
    # For brevity, using itertools.combinations:
    for r in range(len(s) + 1):
        for combo in itertools.combinations(s, r):
            yield combo


def filtered_combos(columns,
                    min_len,
                    max_len,
                    skip_range=None):
    """
    Lazily yield only those combos in the powerset that meet length constraints
    and skip_range constraints.
    """
    for combo in powerset(columns):
        combo_len = len(combo)
        if min_len <= combo_len <= max_len:
            # If skip_range is specified, exclude combos within that length range
            if skip_range is None or not (skip_range[0] <= combo_len <= skip_range[1]):
                yield combo


# Global variable for DataFrame in worker processes
global_df = None

def init_worker(df_shared):
    """
    Initializer function for multiprocessing workers.
    This is called once in each process, so we don't repeatedly pickle df.
    """
    global global_df
    global_df = df_shared


def process_combo_wrapper(combo):
    """
    Wrapper that calls the actual processing function with the global df.
    """
    try:
        return process_combo(combo, global_df)
    except Exception as e:
        # Force a traceback in the child, plus re-raise
        import traceback
        traceback.print_exc()
        raise e


def process_combo(combo, df, verbose=False):
    """
    Build network, fit, compute entropies.
    """
    # Subset of df for these columns
    combo_df = df[list(combo)]

    # ---- FIX #1: Avoid nested parallelism by setting n_jobs=1 ----
    # If 'TreeSearch' or 'estimate' uses parallelization internally,
    # avoid conflict with the outer parallel loop:
    ts = TreeSearch(combo_df, n_jobs=1)
    dag = ts.estimate(estimator_type='chow-liu', show_progress=False)

    model = BayesianNetwork(dag.edges())

    # ---- FIX #2: Fit on the same subset (combo_df) instead of the full df ----
    model.fit(combo_df)
    return [
        f"[{', '.join(combo)}]",
        compute_max_entropy(model),
        compute_shannon_entropy(model),
        compute_collision_entropy_efficient(model),
        compute_min_entropy_efficient(model)
    ]
    

def parallel_combination_entropies(df: pd.DataFrame,
                                   min_combo_length: int = 2,
                                   max_combo_length: int = -1,
                                   skip_range=None,
                                   N_PROCESSES: int = 6,
                                   BATCH_SIZE=5) -> pd.DataFrame:
    """
    Main function that:
      1. Validates min/max combo lengths
      2. Creates a generator for valid combos (powerset with filters)
      3. Spawns a multiprocessing Pool with 'init_worker(df)'
      4. Maps 'process_combo_wrapper' over combos in parallel
      5. Collects results into a DataFrame
    """
    # Validations
    if min_combo_length < 2:
        raise ValueError("min_combo_length must be >= 2!")
    elif min_combo_length > len(df.columns):
        raise ValueError("min_combo_length cannot exceed number of columns!")

    if max_combo_length < 0:
        max_combo_length = len(df.columns)

    logger.info(f"Processing subset lengths {min_combo_length} to {max_combo_length}")

    if skip_range:
        logger.info(f"Skipping subsets between lengths {skip_range[0]} to {skip_range[1]}")

    # Generate combos lazily (no large list in memory)
    combo_iter = filtered_combos(df.columns, min_combo_length, max_combo_length, skip_range)

    # Optional: If you want to BATCH combos, uncomment the chunked approach:
    #
    # from more_itertools import chunked
    # BATCH_SIZE = 1000
    # results = []
    # with Pool(processes=N_PROCESSES, initializer=init_worker, initargs=(df,)) as pool:
    #     for combo_chunk in chunked(combo_iter, BATCH_SIZE):
    #         partial_results = pool.map(process_combo_wrapper, combo_chunk)
    #         results.extend(partial_results)
    #
    # columns = ["Modality", "H_0", "H_1", "H_2", "H_inf"]
    # return pd.DataFrame(results, columns=columns)

    # If you prefer direct mapping over each combo (may be slow for huge combos):
        # 2) Set up the pool
    results = []
    with Pool(processes=N_PROCESSES, initializer=init_worker, initargs=(df,)) as pool:
        for combo_chunk in tqdm(chunked(combo_iter, BATCH_SIZE),
                                desc="Processing combos (unknown total)"):
            # Map each chunk in parallel
            partial_results = pool.map(process_combo_wrapper, combo_chunk)
            # Extend our final results list
            results.extend(partial_results)
            
    columns = ["Modality", "H_0", "H_1", "H_2", "H_inf"]
    return pd.DataFrame(results, columns=columns)
                                     

def entropies_of_sensor_combinations(df: pd.DataFrame,
                                     verbose: bool = False,
                                     all_only: bool = False,
                                     skip_range=None,
                                     min_combo_length: int = 2,
                                     max_combo_length: int = -1) -> pd.DataFrame:
    results = []
    if max_combo_length < 0:
        max_combo_length = len(df.columns)

    # Define filtering based on presence of skip_range
    if skip_range is None:
        # No skip range specified: include all valid combinations
        valid_combos = [ c for c in powerset(df.columns)
            if min_combo_length <= len(c) <= max_combo_length ]
    else:
        # Skip range specified: exclude combinations within the skip range
        valid_combos = [ c for c in powerset(df.columns)
            if (min_combo_length <= len(c) <= max_combo_length) 
            and not (skip_range[0] <= len(c) <= skip_range[1])]

    total_n = len(valid_combos)
    
    if all_only:
        valid_combos = list([df.columns.values])
        total_n = 1
        
    for combo in tqdm(valid_combos, total=total_n, desc="Calculating Joint Entropies"):
        if verbose:
            print("Combo:", combo)
        # Ignore the empty set and single modalities
        if len(combo) <= 1:
            continue

        combo_df = df[list(combo)]
         
        # Estimate a Chow-Liu Bayesian Network
        ts = TreeSearch(combo_df, n_jobs=-2)
        dag = ts.estimate(estimator_type='chow-liu', show_progress=False)
        model = BayesianNetwork(dag.edges())
        model.fit(df) # might have to change this to full df? i.e. to joint_sensors_df, or df as per param
        results.append([f"[{', '.join(combo)}]",
                            compute_max_entropy(model),
                            compute_shannon_entropy(model),
                            compute_collision_entropy_efficient(model),
                            compute_min_entropy_efficient(model)])
    return pd.DataFrame(results, columns=["Modality", "H_0", "H_1", "H_2", "H_inf"])


def display_top_and_bottom_n(joint_df, name=None, n=10, colsort=None):
    if colsort:
        joint_df = joint_df.sort_values(by=colsort, ascending=False)
    display(Markdown(f"## Top {n} Joint Modalities for {name}"))
    display(joint_df[:10])
    display(Markdown(f"## Bottom {n} Joint Modalities for {name}"))
    display(joint_df[-10:])

def plot_distribution_entropies(arr, name, graph_dir, save=True):
    f = plt.figure()
    plt.hist(arr, density=True, alpha=0.4, bins=10, color="k", histtype="stepfilled")  
    plt.xlabel("Min-entropy (bits)")
    plt.ylabel("Density")
    plt.grid()
    if save:
        f.savefig(f"{graph_dir}/{name}_entropies.pdf",bbox_inches="tight")
    plt.close(f)
