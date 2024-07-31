import os
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.axes import Axes
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .analysis import add_bootstrap_methods_to_ols
from .plotting import forest_plot


def run_regression_and_plot(
    data: pd.DataFrame,
    var: str,
    predictors: str,
    num_bootstrap_samples: int = 2000,
    show_plot: bool = True,
    show_summary: bool = True,
    save_fig: bool = True,
    ax: Axes = None,
    figure_kwargs: dict = None,
    forest_plot_kwargs: dict = None,
) -> Tuple[RegressionResultsWrapper, Axes]:
    """
    Run regression, perform bootstrap resampling, and create a forest plot.

    Args:
        data (pd.DataFrame): The data frame containing the variables used
            in the model.
        var (str): The dependent variable in the regression model.
        predictors (str): A string representing predictor variables in the
            regression model.
        num_bootstrap_samples (int, optional): The number of bootstrap
            samples to use for estimating the sampling distribution.
            Defaults to `2000`.
        show_plot (bool, optional): If `True`, shows the plot. Defaults to
            `True`.
        show_summary (bool, optional): If `True`, shows the model summary.
            Defaults to `True`.
        save_fig (bool, optional): If `True`, saves the figure. Defaults to
            `True`.
        ax (Axes, optional): The Axes object to plot on. If `None`, creates
            a new figure and axes. Defaults to `None`.
        figure_kwargs (dict, optional): Keyword arguments to pass to the
            `plt.subplots` function. Defaults to `None`.
        forest_plot_kwargs (dict, optional): Keyword arguments to pass to
            the `forest_plot` function. Defaults to `None`.

    Returns:
        tuple: Containing:
            model
            (statsmodels.regression.linear_model.RegressionResultsWrapper):
                The fitted model object.
            ax (plt.Axes): The Axes object with the plot.
    """
    # 1. Run Regression Analysis
    # smf.ols creates a model from a formula and dataframe
    # f'{var} ~ {predictors}' generates a model formula string.
    # E.g., 'Y ~ X1 + X2'
    model = smf.ols(f"{var} ~ {predictors}", data=data).fit()

    # Display the model summary, which includes things like coefficient values,
    # R-squared, p-values, etc.
    if show_summary:
        print(model.summary())

    # 2. Bootstrap
    # add_bootstrap_methods_to_ols adds bootstrap methods to the model.
    fitted_model = add_bootstrap_methods_to_ols(model)

    # Perform bootstrap resampling to estimate the sampling distribution of the
    # estimator and calculate confidence intervals
    fitted_model.bootstrap(num_bootstrap_samples)

    # 3. Create Forest Plot
    # This creates a new figure and axes object for the plot
    if figure_kwargs is None:
        figure_kwargs = {}
    if ax is None:
        f, ax = plt.subplots(**figure_kwargs)

    # Create the forest plot using the forest_plot function
    if forest_plot_kwargs is None:
        forest_plot_kwargs = {}
    forest_plot(fitted_model, ax=ax, **forest_plot_kwargs)

    # 4. Save and/or Display the Plot
    # If save_fig is True, save the figure to disk before showing it
    if save_fig:
        # Create a file name string, replacing spaces and plus signs with
        # underscores and removing pipe characters
        predictors_str = (
            predictors.replace(" + ", "_").replace(" ", "").replace("|", "or")
        )

        # Create a filename using both dependent and predictor variable names,
        # improving traceability of the saved plot
        fig_name = f"{var}_vs_{predictors_str}.svg"

        # Check if 'figures' directory exists. If not, create it.
        if not os.path.exists("figures"):
            os.makedirs("figures")

        # Adjust the layout to ensure that it looks good and then save the
        # figure as an SVG file in the 'figures' directory
        plt.tight_layout()
        plt.savefig(f"figures/{fig_name}")

    # Show the plot in the notebook/output
    if not show_plot:
        plt.close()

    # 5. Return Axes This will return the ax object, which can be used for
    # further customization outside of the function
    return model, ax
