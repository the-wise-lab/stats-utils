from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_grouped_corr_heatmap(
    data: pd.DataFrame,
    subplot_kwargs: Dict = {},
    heatmap_kwargs: Dict = {},
    axis_labels: str = None,
    grouping_linewidth: float = 2,
    ax: plt.Axes = None,
    **kwargs: Any
) -> plt.Axes:
    """
    Plot a correlation heatmap with grouped variable labels.

    Args:
        data (pd.DataFrame): DataFrame with variables named in the format
            "group_variable".
        subplot_kwargs (Dict, optional): Keyword arguments to pass to
            `plt.subplots()`. Defaults to `{}`.
        heatmap_kwargs (Dict, optional): Keyword arguments to pass to
            `sns.heatmap()`. Defaults to `{}`.
        axis_labels (str, optional): Label for the x and y axes. Defaults
            to `None`.
        grouping_linewidth (float, optional): Width of the lines separating
            variable groups. Defaults to `2`.
        ax (plt.Axes, optional): Axes to plot on. Defaults to `None`.

    Returns:
        plt.Axes: The created axes.

    Example:
        ax = plot_grouped_corr_heatmap(data)
        plt.savefig('../correlation_matrix.png', dpi=`300`)
    """

    # Set default kwargs
    subplot_kwargs.setdefault("figsize", (11, 9))
    heatmap_kwargs.setdefault("cmap", "RdBu")
    heatmap_kwargs.setdefault("center", 0)
    heatmap_kwargs.setdefault("square", True)
    heatmap_kwargs.setdefault("linewidths", 0.5)
    heatmap_kwargs.setdefault("cbar_kws", {"shrink": 0.5})
    heatmap_kwargs.setdefault("linecolor", "black")

    # Axis labels
    if axis_labels is None:
        axis_labels = "Variable group"

    # Compute the correlation matrix
    corr = data.corr()

    # Set up the matplotlib figure
    if ax is None:
        f, ax = plt.subplots(**subplot_kwargs)

    # Draw the heatmap
    sns.heatmap(corr, ax=ax, **heatmap_kwargs)

    # Draw border round outside of heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)

    # Define groups and their labels dynamically
    variables = data.columns
    # Define groups and their labels dynamically
    groups = {
        key: [min(idx), max(idx) + 1]
        for key, idx in pd.Series(
            {var: idx for idx, var in enumerate(variables)}
        )
        .groupby(lambda x: x.split("_")[0])
        .agg(list)
        .to_dict()
        .items()
    }
    labels = list(groups.keys())

    # Set x and y ticks to center labels
    ax.set_xticks([np.mean(range(*rg)) + 0.5 for rg in groups.values()])
    ax.set_yticks([np.mean(range(*rg)) + 0.5 for rg in groups.values()])

    # Label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )
    plt.setp(
        ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor"
    )

    # Draw gridlines to separate variable groups
    for group_range in groups.values():
        ax.axvline(
            x=group_range[1], color="black", linewidth=grouping_linewidth
        )
        ax.axhline(
            y=group_range[1], color="black", linewidth=grouping_linewidth
        )

    # Set colorbar title
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Correlation")

    # Set titles and labels
    ax.set_xlabel(axis_labels)
    ax.set_ylabel(axis_labels)

    plt.tight_layout()

    return ax
