import matplotlib.pyplot as plt
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx


def build_graph_from_scores(factor_scores: List[pd.DataFrame]) -> nx.DiGraph:
    """Build a directed graph based on correlations between factor scores.

    Args:
        factor_scores (List[pd.DataFrame]): A list of DataFrames containing factor scores.

    Returns:
        nx.DiGraph: A directed graph (DiGraph) representing the correlations.

    Example:
        G = build_graph_from_scores([df1, df2])
    """

    # Check whether NetworkX is installed (required for this function, but not a dependency of stats_utils)
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "The plot_factor_correlations function requires NetworkX. Please install it using pip or conda."
        )

    G = nx.DiGraph()

    # Iterate over the list of factor scores dataframes
    for i in range(1, len(factor_scores)):
        df1 = factor_scores[i - 1][
            [col for col in factor_scores[i - 1].columns if "ML" in col]
        ]
        df2 = factor_scores[i][[col for col in factor_scores[i].columns if "ML" in col]]

        # Iterate over the columns in df1
        for n, j in enumerate(df1.columns):
            # Generate a unique node ID for the current column in df1
            node_id = int(((i - 1) * i / 2) + n)

            # If node is not already in the graph, add it with a 'layer' attribute
            if node_id not in G.nodes():
                G.add_node(node_id, layer=i - 1)

            # Iterate over the columns in df2
            for m, k in enumerate(df2.columns):
                # Generate a unique subnode ID for the current column in df2
                subnode_id = int((i * (i + 1) / 2) + m)

                # If subnode is not already in the graph, add it with a 'layer' attribute
                if subnode_id not in G.nodes():
                    G.add_node(subnode_id, layer=i)

                # Add an edge between the node and subnode with correlation as weight
                G.add_edge(node_id, subnode_id, weight=df1[j].corr(df2[k]))

    return G


def filter_graph_by_threshold(G: nx.DiGraph, threshold: float) -> nx.DiGraph:
    """Remove edges from a graph based on a threshold value.

    Args:
        G (nx.DiGraph): A directed graph (DiGraph).
        threshold (float): Minimum weight value to keep an edge in the graph.

    Returns:
        nx.DiGraph: A new filtered directed graph.

    Example:
        G_filtered = filter_graph_by_threshold(G, 0.5)
    """

    # Create a copy of the original graph
    G_filtered = G.copy()

    # Identify edges with weights below the threshold
    edges_to_remove = [
        (u, v) for u, v, d in G_filtered.edges(data=True) if d["weight"] <= threshold
    ]

    # Remove the identified edges from the copied graph
    G_filtered.remove_edges_from(edges_to_remove)

    return G_filtered


def plot_factor_correlations(
    factor_scores: List[pd.DataFrame],
    threshold: float = 0.5,
    labels: Dict[int, str] = None,
    figure_kwargs: Any = {},
    node_plot_kwargs: Any = {},
    labels_plot_kwargs: Any = {},
    edges_plot_kwargs: Any = {},
    edge_labels_plot_kwargs: Any = {},
) -> None:
    """Create and plot a graph representation of correlations between factor scores at multiple
    levels of a hierarchy.

    Args:
        factor_scores (List[pd.DataFrame]): A list of DataFrames containing factor scores. Factor scores
        are expected to be in the columns of the DataFrames, with the format ML*, where * is the factor number.
        threshold (float, optional): Minimum correlation value to consider for plotting. Defaults to 0.5.
        labels (Dict[int, str], optional): Dictionary mapping node indices to custom labels.
            Default is None, meaning no custom labels are provided.
        figure_kwargs (Any, optional): Keyword arguments to pass to plt.figure. Defaults to {}.
        node_plot_kwargs (Any, optional): Keyword arguments to pass to nx.draw_networkx_nodes.
            Defaults to {}.
        labels_plot_kwargs (Any, optional): Keyword arguments to pass to nx.draw_networkx_labels.
            Defaults to {}.
        edges_plot_kwargs (Any, optional): Keyword arguments to pass to nx.draw_networkx_edges.
            Defaults to {}.
        edge_labels_plot_kwargs (Any, optional): Keyword arguments to pass to nx.draw_networkx_edge_labels.
            Defaults to {}.


    Returns:
        None: This function returns nothing but creates a matplotlib plot.

    Example:
        plot_factor_correlations([df1, df2], threshold=0.6, labels={1: 'L1: F1', 2: 'L2: F1', 3: 'L2: F2})
    """

    # Check whether NetworkX is installed (required for this function, but not a dependency of stats_utils)
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "The plot_factor_correlations function requires NetworkX. Please install it using pip or conda."
        )

    G = build_graph_from_scores(factor_scores)
    G_filtered = filter_graph_by_threshold(G, threshold)

    # Extract edge labels based on the weights
    edge_labels = {
        (u, v): f"{d['weight']:.2f}" for u, v, d in G_filtered.edges(data=True)
    }

    # Normalize edge weights for plotting
    weights = [d["weight"] for _, _, d in G_filtered.edges(data=True)]
    normalized_weights = (weights - np.min(weights)) / (
        np.max(weights) - np.min(weights)
    )
    normalized_weights = (normalized_weights * 0.7) + 0.3

    # Create a color map for the edges based on their normalized weights
    cmap = plt.get_cmap("Greys")
    edge_colors = [cmap(weight) for weight in normalized_weights]

    # Define node positioning based on layers using multipartite_layout
    pos = nx.multipartite_layout(G_filtered, subset_key="layer", align="vertical")

    # Set figure size and plot details
    figure_kwargs.setdefault("figsize", (6, 6))
    plt.figure(**figure_kwargs)

    # Draw nodes with labels
    # Set default kwargs for nodes and labels
    node_plot_kwargs.setdefault("node_size", 2000)
    node_plot_kwargs.setdefault("node_color", "#2782e3")
    node_plot_kwargs.setdefault("alpha", 0.0)
    node_plot_kwargs.setdefault("node_shape", "s")
    labels_plot_kwargs.setdefault("font_size", 10)
    labels_plot_kwargs.setdefault("font_color", "white")
    labels_plot_kwargs.setdefault("bbox", dict(boxstyle="square,pad=0.3", alpha=0.7))

    nx.draw_networkx_nodes(G_filtered, pos, **node_plot_kwargs)
    nx.draw_networkx_labels(G_filtered, pos, labels=labels, **labels_plot_kwargs)

    # Draw edges with colors based on their weights
    # Set default kwargs for edges
    edges_plot_kwargs.setdefault("width", 3)
    nx.draw_networkx_edges(
        G_filtered,
        pos,
        edge_color=edge_colors,
        node_size=node_plot_kwargs["node_size"],
        **edges_plot_kwargs,
    )

    # Annotate edges with their weights (correlation values)
    # Set default kwargs for edge labels
    edge_labels_plot_kwargs.setdefault("font_size", 8)
    edge_labels_plot_kwargs.setdefault("rotate", False)
    edge_labels_plot_kwargs.setdefault("label_pos", 0.6)
    nx.draw_networkx_edge_labels(
        G_filtered, pos, edge_labels=edge_labels, **edge_labels_plot_kwargs
    )

    plt.title("Correlations between factor scores across levels")


def plot_factor_loadings(
    loadings_file: str,
    factor_labels: List[str],
    cmap: LinearSegmentedColormap = None,
    ax: plt.axes = None,
    figure_kwargs: dict = None,
    adjust_spacing: bool = True,
) -> List[plt.axes]:
    """
    Reads a loadings file and reshapes the data for visualization of factor loadings across multiple measures.

    The loadings file should be a CSV containing factor analysis loadings with the following columns:
        - 'itemNumber': An integer identifier for each item.
        - 'measure': A string indicating the measure or test each item belongs to.
        - Any number of columns prefixed with 'ML' which contain the loading values for corresponding factors.

    The function melts the loadings data into a long format suitable for plotting, with each 'ML' prefixed column
    being treated as a separate factor. It also sets up a custom colormap for the plots if one is not provided and
    creates a bar plot for each factor's loadings on different items, separated by measure. Each subplot corresponds
    to one factor and displays item loadings across all measures.

    Args:
        loadings_file (str): Path and filename of the factor loadings CSV file.
        factor_labels (List[str]): List of labels to use for each factor.
        cmap (LinearSegmentedColormap, optional): Custom colormap to use for plotting. Defaults to None.
        ax (plt.axes, optional): Axes to plot on. Defaults to None.
        figure_kwargs (dict, optional): Keyword arguments to pass to plt.subplots. Defaults to None.
        adjust_spacing (bool, optional): Whether to adjust the spacing between subplots. Defaults to True.
    Returns:
        matplotlib.axes._subplots.AxesSubplot: The resulting plot.

    Example:
        axes = plot_factor_loadings("loadings.csv", ['Internalising', 'Externalising', 'Inattentive', 'Withdrawal'])
    """

    # Load the factor loadings
    loadings = pd.read_csv(loadings_file)

    # Reshape data for plotting
    loadings_long = pd.melt(
        loadings.drop(["item"], axis=1),
        id_vars=["itemNumber", "measure"],
        value_vars=[i for i in loadings.columns if "ML" in i],
        var_name="factor",
        value_name="loading",
    )
    loadings_long["itemNumber"] = loadings_long["itemNumber"].astype(int)

    # Define the custom colormap if not given
    if cmap is None:
        cmap_colours = ["#FF00AA", "#B123C4", "#0074FF", "#00F8FF"]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", cmap_colours)

    # Create figure
    # Set default figure kwargs
    figure_kwargs = figure_kwargs or {}
    figure_kwargs.setdefault("figsize", (8, 8))
    if ax is None:
        f, ax = plt.subplots(len(loadings_long["factor"].unique()), 1, **figure_kwargs)
    else:
        # make sure ax is as long as the number of factors
        if len(ax) != len(loadings_long["factor"].unique()):
            raise ValueError(
                "The number of axes provided does not match the number of factors."
            )

    # Get the unique measures and create a color map
    unique_measures = loadings_long["measure"].unique()
    measure_labels = [i.replace("_", " ") for i in unique_measures]

    for n, factor in enumerate(sorted(loadings_long["factor"].unique())):
        ax[n].axhline(0, color="gray", linewidth=1, linestyle=":")
        # Plot bars for each measure
        for idx, measure in enumerate(unique_measures):
            measure_data = loadings_long[
                (loadings_long["factor"] == factor)
                & (loadings_long["measure"] == measure)
            ]
            ax[n].bar(
                measure_data["itemNumber"],
                measure_data["loading"],
                label=measure_labels[idx],
                color=cmap(idx / len(unique_measures)),
            )
        # Set labels and title
        ax[n].set_xlabel("Item Number")
        ax[n].set_ylabel("Loading")
        ax[n].set_title(factor_labels[n], fontweight="medium")
        ax[n].set_xlim(0, loadings_long["itemNumber"].max() + 1)

        # Remove spines
        sns.despine(ax=ax[n])
        
    # Add legend
    ax[n].legend(title="Measure", loc="center right", bbox_to_anchor=(1.2, 2), ncol=1)
    # Add spacing between subplots
    if adjust_spacing:
        plt.subplots_adjust(hspace=0.7)

    return ax


def plot_scree(
    eigenvalues: List[float],
    n_factors: int = 30,
    show_threshold: bool = True,
    ax: plt.axes = None,
    figure_kwargs: Dict = None,
    scatter_kwargs: Dict = None,
) -> None:
    """
    Plots a scree plot for the provided eigenvalues with customizable figure and scatter properties.

    This function creates a scree plot which is helpful to visually assess
    the number of factors to retain in an exploratory factor analysis (EFA).
    It plots the eigenvalues against the number of factors and draws a
    horizontal line at y=1 for reference.

    Args:
    eigenvalues (List[float]): A list of eigenvalues from a factor analysis.
    n_factors (int, optional): The number of factors to consider for the plot. Default is 30.
    show_threshold (bool, optional): Whether to show a horizontal line at y=1. Default is True.
    ax (plt.axes, optional): Axes to plot on. Defaults to None.
    figure_kwargs (Dict, optional): Keyword arguments to be passed to plt.figure.
                                    Defaults are {'figsize': (3.5, 2.3), 'dpi': 150}.
    scatter_kwargs (Dict, optional): Keyword arguments to be passed to plt.scatter.
                                     Defaults are {'s': 10}.

    Returns:
    None: The function plots the scree plot but does not return any value.
    """

    # Ensure kwargs are not None
    figure_kwargs = figure_kwargs or {}
    scatter_kwargs = scatter_kwargs or {}

    # Set default values if not provided
    figure_kwargs.setdefault("figsize", (3.5, 2.3))
    figure_kwargs.setdefault("dpi", 150)
    scatter_kwargs.setdefault("s", 10)

    # Create figure with kwargs
    if ax is None:
        fig, ax = plt.subplots(1, 1, **figure_kwargs)

    # Plot a horizontal line at y=1 for reference
    if show_threshold:
        ax.axhline(y=1, color="gray", linestyle=":")

    # Scatter plot of eigenvalues against the range of factors with kwargs
    ax.scatter(range(1, n_factors + 1), eigenvalues[:n_factors], **scatter_kwargs)

    # X and Y axis labels
    ax.set_xlabel("Factor")
    ax.set_ylabel("Eigenvalue")
