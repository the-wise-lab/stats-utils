import os
from typing import List, Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def plot_posteriors(
    results: az.InferenceData,
    var_names: List[str],
    titles: List[str],
    file_name: Optional[str] = None,
    **kwargs
) -> None:
    """
    Creates a summary of the Bayesian results and plots the posterior distributions with optional keyword
    arguments for plotting functions. Optionally saves the figure to a file.

    Args:
        results: The results object from the Bayesian model fit, typically an `InferenceData` object.
        var_names: A list of variable names to summarize and plot.
        titles: A list of titles corresponding to the variables. Each title corresponds to one variable name.
        file_name: Optional; The name of the file to save the plot. If None, the plot is not saved.
        **kwargs: Optional keyword arguments passed to Seaborn KDE plot function.

    Returns:
        None: This function does not return anything.

    Example:
        >>> plot_bayesian_results(results_2, ["mu", "tau", "eta"], ["Mean", "Standard Deviation", "Noise"],
                                  file_name="my_model_posteriors.svg", linewidth=2.5, linestyle='--')
    """
    # Creating a summary for the specified variables
    summary = az.summary(
        results,
        var_names=var_names,
        hdi_prob=0.95,
    )

    # Creating a figure and axes objects
    f, ax = plt.subplots(1, len(var_names), figsize=(len(var_names) * 2, 2.2))

    # Retrieving the default colour palette from matplotlib
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Initializing a colour list to store colors for each distribution plot
    colours = [palette[0]] * len(var_names)

    # Determine the colour of the distribution based on 95% HDI
    for i, v in enumerate(var_names):
        var_summary = summary.loc[v]
        if np.sign(var_summary["hdi_2.5%"]) == np.sign(var_summary["hdi_97.5%"]):
            colours[i] = palette[1]

    # Plotting distributions for each variable
    for i, v in enumerate(var_names):
        ax[i].set_xlabel(r"$\beta$ value")  # Label x-axis
        ax[i].axvline(
            0, color="black", linestyle="--", alpha=0.5
        )  # Add a vertical line at x=0
        sns.kdeplot(
            results.posterior[v].values.flatten(),
            fill=True,
            ax=ax[i],
            color=colours[i],
            **kwargs  # Pass additional kwargs to seaborn kdeplot
        )
        ax[i].set_ylabel("" if i > 0 else "Posterior\ndensity")
        ax[i].set_title(titles[i])  # Title

    sns.despine()

    plt.tight_layout()

    # Save the figure if file_name is provided
    if file_name:
        figure_folder = "figures"
        if not os.path.exists(figure_folder):
            os.makedirs(figure_folder)
        plt.savefig(os.path.join(figure_folder, file_name))


def plot_posterior_forest(
    results: az.InferenceData,
    var_names: List[str],
    titles: List[str],
    credible_interval: float = 0.95,
    file_name: Optional[str] = None,
    **kwargs
) -> None:
    """
    Creates a forest plot of the Bayesian results with point estimates and credible intervals,
    with variables displayed along the x-axis.

    Args:
        results: The results object from the Bayesian model fit, typically an `InferenceData` object.
        var_names: A list of variable names to summarize and plot.
        titles: A list of titles corresponding to the variables. Each title corresponds to one variable name.
        credible_interval: The credible interval to use for the error bars (default is 0.95).
        file_name: Optional; The name of the file to save the plot. If None, the plot is not saved.
        **kwargs: Optional keyword arguments passed to the plotting functions.

    Returns:
        None: This function does not return anything.

    Example:
        >>> plot_forest_bayesian_results(results_2, ["mu", "tau", "eta"],
                                         ["Mean", "Standard Deviation", "Noise"],
                                         file_name="my_model_forest_plot.svg")
    """
    # Creating a summary for the specified variables
    summary = az.summary(
        results, var_names=var_names, hdi_prob=credible_interval, kind="stats"
    )

    # Get default colour palette from matplotlib
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Creating a figure
    fig, ax = plt.subplots(figsize=(3, 2.5))

    # Initializing a colour list to store colors for each distribution plot
    colours = [palette[0]] * len(var_names)

    # Plotting points with error bars for each variable
    for i, v in enumerate(var_names):
        # Extracting the point estimate and the HDI
        mean = summary.loc[v, "mean"]
        hdi_low = summary.loc[v, "hdi_2.5%"]
        hdi_high = summary.loc[v, "hdi_97.5%"]

        # Get colour
        if np.sign(hdi_low) == np.sign(hdi_high):
            colours[i] = palette[1]

        # Plotting the point estimate
        ax.scatter(
            i,
            mean,
            color=colours[i],
            edgecolors="black",
            linewidths=1,
        )

        # Plotting the error bars (credible interval)
        ax.errorbar(
            x=[i, i],
            y=[hdi_low, hdi_high],
            color=colours[i],
            capsize=0,
            capthick=2,
            linewidth=2,
            ecolor=colours[i],
            zorder=-1,
            alpha=0.8,
        )

    # Setting the x-axis to show the variable titles
    ax.set_xticks(np.arange(len(var_names)))

    # Replace spaces with newlines in titles
    titles = [title.replace(" ", "\n") for title in titles]
    ax.set_xticklabels(titles, rotation=45, ha="right")

    ax.set_ylabel("Parameter estimate")
    ax.axhline(
        0, color="grey", linestyle="--", alpha=0.7
    )  # Add a horizontal line at zero for reference

    sns.despine()
    plt.tight_layout()

    # Save the figure if file_name is provided
    if file_name:
        if not os.path.exists("figures"):
            os.makedirs("figures")
        plt.savefig(os.path.join("figures", file_name), bbox_inches="tight")

    plt.show()
