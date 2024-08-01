from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .analysis import ModelOutput


def plot_r2s(
    model_output: ModelOutput,
    save_path: Union[str, None] = None,
    ax: plt.Axes = None,
    show_ylabel: bool = True,
) -> None:
    """
    Function to plot the adjusted r2s for each model fit in a sequential
    regression.

    Args:
        model_output (ModelOutput): Model output from sequential_regression
            function.
        save_path (Union[str, None], optional): Path to save figure to.
            Defaults to `None`.
        ax (plt.Axes, optional): Axis to plot on. If `None`, will create a new
            figure. Defaults to `None`.
        show_ylabel (bool, optional): Whether to show the y label. Defaults
            to `True`.
    """

    cmap_colours = ["#FF00AA", "#B123C4", "#0074FF"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", cmap_colours)

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2.4, 2.2))

    x = np.arange(model_output.n_solutions + 1)
    y = model_output.r2s

    # interpolate x and y
    x_interp = np.linspace(0, model_output.n_solutions, 100)
    y_interp = np.interp(
        x_interp, np.arange(model_output.n_solutions + 1), model_output.r2s
    )

    # Create a set of points (x, y) as a 2D array
    points = np.array([x_interp, y_interp]).T.reshape(-1, 1, 2)

    # Create a set of segments as a 3D array
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a normalized color map for the line gradient
    norm = plt.Normalize(x_interp.min(), x_interp.max())

    # Create a LineCollection with the segments and colors
    lc = LineCollection(segments, colors=cmap(norm(x_interp)), linewidth=3)

    # ax.plot(model_outputs['p_correct'].r2s, 'o-')
    ax.scatter(x, y, c=cmap(norm(x)), s=60)

    # Add the LineCollection to the axis
    ax.add_collection(lc)

    # ax.set_title(p.replace('_', ' '))
    ax.set_title(model_output.y_var.replace("_", " "))
    ax.set_xlabel("Number of factors")

    if show_ylabel:
        ax.set_ylabel(r"Adjusted $R^2$")

    # Add significance stars
    for j in range(1, model_output.n_solutions + 1):
        if model_output.anova_results["Pr(>F)"][j] < 0.05:
            star_string = "*"
            if model_output.anova_results["Pr(>F)"][j] < 0.01:
                star_string = "**"
            if model_output.anova_results["Pr(>F)"][j] < 0.001:
                star_string = "***"
            ypos = ax.get_ylim()[1] * 1
            ax.text(
                j - 0.1,
                ypos,
                star_string,
                fontsize=15,
                fontweight="regular",
                horizontalalignment="center",
                color="#292929",
            )

    # set x tick labels to be integers from 0 to n_solutions
    ax.set_xticks(np.arange(model_output.n_solutions + 1))

    # make axis tick font regular weight
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight("regular")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight("regular")

    # extend y lim
    ax.set_ylim(
        [
            ax.get_ylim()[0],
            ax.get_ylim()[1] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.3,
        ]
    )

    ax.set_xlim(-0.5, model_output.n_solutions + 0.5)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)


def forest_plot(
    model: RegressionResultsWrapper,
    alpha: float = 0.05,
    ax: plt.Axes = None,
    rename_dict: dict = None,
    capitalise_vars: bool = True,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    exclude_param_names: list = None,
    significance_thresholds: Dict[float, str] = {
        0.05: "*",
        0.01: "**",
        0.001: "***",
    },
) -> None:
    """
    Creates a forest plot of the betas with 95% confidence intervals and
    significance stars.

    Args:
        model (RegressionResultsWrapper): A fitted regression model from
            statsmodels.
        alpha (float, optional): The significance level for the confidence
            intervals and significance stars. Defaults to `0.05`.
        ax (plt.Axes, optional): The axes to plot on. If `None`, a new figure
            and axes will be created. Defaults to `None`.
        rename_dict (dict, optional): A dictionary of variable names to rename.
            Defaults to `None`.
        capitalise_vars (bool, optional): Whether to capitalise the variable
            names. Defaults to `True`.
        show_xlabel (bool, optional): Whether to show the x label. Defaults
            to `True`.
        show_ylabel (bool, optional): Whether to show the y label. Defaults
            to `True`.
        exclude_param_names (list, optional): A list of parameter names to
            exclude from the plot. Defaults to `None`.
        significance_thresholds (Dict[float, str], optional): A dictionary of
            significance thresholds and the corresponding significance stars.
            Defaults to `{0.05: "*", 0.01: "**", 0.001: "***"}`.

    Returns:
        None: The function does not return any value. It displays the forest
            plot using Matplotlib.

    Examples:
        ```python
        import statsmodels.api as sm
        from stats_utils.regression import forest_plot

        # Load the data
        data = sm.datasets.get_rdataset("mtcars", "datasets").data

        # Fit a linear regression model
        model = sm.OLS(data["mpg"], sm.add_constant(data["wt"])).fit()

        # Create a forest plot
        forest_plot(model)
        ```
    """

    # Get default colours
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Get boolean array of which parameters to exclude
    if exclude_param_names is None:
        exclude_param_names = []
    include = np.array(
        [
            param_name not in exclude_param_names
            for param_name in model.params.index[1:]
        ]
    )

    # Get the coefficients and their standard errors
    coefs = model.params[1:][include]
    # conf_intervals = model.conf_int(alpha=alpha)[1:]
    conf_intervals = model.conf_int_bootstrap(alpha=alpha)[1:]
    lower_bounds = conf_intervals[0][include]
    upper_bounds = conf_intervals[1][include]
    p_values = model.pvalues_bootstrap[1:][include]

    # Calculate the errors (distance from the coefficient to the bounds)
    errors_upper = upper_bounds - coefs
    errors_lower = coefs - lower_bounds

    # Number of coefficients
    n_coef = len(coefs)

    # Set up the plot
    if ax is None:
        f, ax = plt.subplots(figsize=(n_coef * 0.4, 1.8))

    # Determine the colour of each point based on the p-value
    colours = [
        default_colors[1] if p_value < alpha else default_colors[0]
        for p_value in p_values
    ]

    # Plot the coefficients
    # Errorbars first
    ax.errorbar(
        x=np.arange(n_coef),
        y=coefs,
        yerr=[errors_lower, errors_upper],
        fmt="none",
        capsize=0,
        capthick=2,
        linewidth=2,
        ecolor=colours,
        zorder=-1,
        alpha=0.8,
    )

    # Then the points
    ax.scatter(
        x=np.arange(n_coef),
        y=coefs,
        s=50,
        marker="o",
        color=colours,
        edgecolors="black",
        linewidths=1,
    )

    # Add significance stars
    for i, (coef, _, p_value) in enumerate(zip(coefs, errors_upper, p_values)):
        if p_value < alpha:
            # Get the significance star string
            star_string = ""
            # Start from the highest threshold
            for threshold in sorted(
                significance_thresholds.keys(), reverse=True
            ):
                # If the p-value is lower than the threshold, add the
                # corresponding significance star
                if p_value < threshold:
                    star_string = significance_thresholds[threshold]
            # Add the star
            ax.text(
                i,
                coef + errors_upper.iloc[i],
                star_string,
                ha="center",
                va="bottom",
                fontsize=14,
                color="#292929",
            )

    # Rename the variables
    if rename_dict is not None:
        var_names = [
            rename_dict[var] if var in rename_dict else var
            for var in coefs.index
        ]
    else:
        var_names = coefs.index

    # Capitalise the variable names
    if capitalise_vars:
        var_names = [var[0].upper() + var[1:] for var in var_names]

    # Set the x-ticks and their labels
    ax.set_xticks(np.arange(n_coef), var_names, rotation=45, ha="right")

    # make axis tick font regular weight
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight("regular")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight("regular")

    # Add a horizontal line at y=0
    ax.axhline(y=0, linestyle=":", color="gray", linewidth=1)

    # Set the plot's title, xlabel, and ylabel
    if show_xlabel:
        ax.set_xlabel("Variable")
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(r"$\beta$ (+/- SE)")
    else:
        ax.set_ylabel("")

    # Adjust Y-axis limits
    ax.set_ylim(
        min(lower_bounds) - max(errors_lower) * 0.5,
        max(upper_bounds) + max(errors_upper) * 2,
    )

    # Set the x-axis limits
    ax.set_xlim(-0.5, n_coef - 0.5)

    # Remove the top and right spines
    sns.despine()
