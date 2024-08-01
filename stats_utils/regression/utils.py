import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.axes import Axes
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .analysis import add_bootstrap_methods_to_ols
from .plotting import forest_plot
from ..utils import dataframe_to_markdown


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
    This function performs three steps:

    1) Run a regression analysis using the `statsmodels` package.\n
    2) Perform bootstrap resampling to estimate confidence intervals
         and p-values.\n
    3) Create a forest plot of the regression coefficients (indicating
        significance and confidence intervals).\n

    > ⚠️ NOTE: This function estimates confidence intervals and p-values
    using boostrapping. The p-values and confidence intervals given in
    the model summary are not derived from the bootstrap samples and so
    may not necessarily correspond to what is shown in the figure (which
    uses confidence intervals and significance derived from resampling).
    To obtain p-values and confidence intervals from the bootstrap samples,
    you can access them from the `model` object returned by this
    function. For example, to get the 95% confidence intervals for all
    predictors, you can use `model.conf_int_bootstrap(alpha=0.05)`,
    and to get the p-values, you can use `model.pvalues_bootstrap`.

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

    Example:
        ```python
        # Fit model and plot
        model, ax = run_regression_and_plot(
            data,
            "Y",
            "X1 + X2 + X3",
        )

        # Get the 95% confidence intervals for the predictors
        print(model.conf_int_bootstrap(alpha=0.05))

        # Get the p-values for the predictors
        print(model.pvalues_bootstrap)
        ```


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


def ols_to_markdown_table(
    ols_result: RegressionResultsWrapper,
    use_bootstrapping: bool = True,
    predictor_rename_dict: Optional[Dict[str, str]] = None,
    exclude_predictors: Optional[List[str]] = [],
    column_rename_dict: Optional[Dict[str, str]] = None,
    round_dict: Optional[Dict[str, int]] = None,
    alpha_corr: float = None,
) -> str:
    """
    Convert the summary table of an OLS regression result to a markdown table.
    The intercept is dropped from the output.

    Args:
        ols_result (sm.RegressionResultsWrapper):
            The result object of an OLS regression.
        use_bootstrapping (bool, optional): Whether to replace CIs and p-values
            with bootstrapped values if available. Defaults to `True`.
        predictor_rename_dict (Optional[Dict[str, str]], optional): A
            dictionary to rename the predictors in the summary table. If not
            included, predictors will be tidied slightly instead. Defaults to
            `None`.
        exclude_predictors (Optional[List[str]], optional): A list of
            predictors to exclude from the summary table. Defaults to `[]`.
        column_rename_dict (Optional[Dict[str, str]], optional): A
            dictionary to rename the summary table columns. Defaults to a
            pre-specified dictionary if not provided.
        round_dict (Optional[Dict[str, int]], optional): A dictionary to set
            the rounding precision for each column. Defaults to a pre-specified
            dictionary if not provided.
        alpha_corr (float, optional): The alpha level for multiple comparison
            correction. If provided, the p-values will be corrected using the
            Holm-Bonferroni method and a new column will be added to the table
            with the corrected p-values (i.e., multiplied by
            0.05 / alpha_corr). Defaults to `None`.

    Returns:
        str: The markdown table representing the summary table of the OLS
            regression result.

    Example:
        ```python
        # Fit model
        model = smf.ols("Y ~ X1 + X2", data).fit()

        # Convert summary table to markdown
        markdown_table = ols_to_markdown_table(model)
        ```
    """

    # Default column renaming dict
    if column_rename_dict is None:
        column_rename_dict = {
            "Coef.": "$\\beta$",
            "Std.Err.": "$\\beta_{SE}$",
            "t": "$t$",
            "P>|t|": "$p$",
            "[0.025": "$CI_{2.5}$",
            "0.975]": "$CI_{97.5}$",
        }

    # Default rounding precision dict
    if round_dict is None:
        round_dict = {
            "$\\beta$": 2,
            "$\\beta_{SE}$": 2,
            "$t$": 2,
            "$p$": 3,
            "$p_{corr}$": 3,
            "$CI_{2.5}$": 2,
            "$CI_{97.5}$": 2,
        }

    # Convert the summary table to a DataFrame
    summary_df = pd.DataFrame(ols_result.summary2().tables[1])

    # Rename the columns
    summary_df = summary_df.rename(columns=column_rename_dict)

    # Replace CIs and pvals with bootstrapped values if bootstrapping was used
    if use_bootstrapping:
        # Check whether the model has model.pvalues_bootstrap attribute
        if hasattr(ols_result, "pvalues_bootstrap"):
            # Replace p-values with bootstrapped values
            summary_df["$p$"] = ols_result.pvalues_bootstrap
            # Replace CI values with bootstrapped values
            summary_df["$CI_{2.5}$"] = ols_result.conf_int()[0]
            summary_df["$CI_{97.5}$"] = ols_result.conf_int()[1]
            # Remove the t-values column
            summary_df = summary_df.drop(columns="$t$")
        else:
            print(
                "OLS model does not have pvalues_bootstrap attribute. "
                "Using original values."
            )

    # Rename the predictors
    # If we don't have a rename dict, tidy the predictor names slightly
    if predictor_rename_dict is None:
        # Replace __ with space
        summary_df.index = summary_df.index.str.replace("__", " ")
        # Replace _ with space
        summary_df.index = summary_df.index.str.replace("_", " ")
        # Capitalize the first letter of each word
        summary_df.index = summary_df.index.str.title()
    else:
        # Rename the predictors using the provided dictionary
        summary_df = summary_df.rename(index=predictor_rename_dict)

    # Drop the intercept row
    summary_df = summary_df.drop(index="Intercept")

    # Drop any excluded predictors
    summary_df = summary_df.drop(index=exclude_predictors, errors="ignore")

    # Correct p-values for multiple comparisons
    if alpha_corr is not None:
        summary_df["$p_{corr}$"] = summary_df["$p$"] * (0.05 / alpha_corr)
        # Make sure values don't go above 1
        summary_df["$p_{corr}$"] = summary_df["$p_{corr}$"].clip(upper=1)
        # Put the corrected p-value column next to the original p-value column
        correct_col_order = [
            "$\\beta$",
            "$\\beta_{SE}$",
            "$t$",
            "$p$",
            "$p_{corr}$",
            "$CI_{2.5}$",
            "$CI_{97.5}$",
        ]
        # Reorder the columns, if each column is in the dataframe
        summary_df = summary_df[
            [col for col in correct_col_order if col in summary_df.columns]
        ]

    # Remove columns that arne't needed from the rounding dict
    round_dict = {
        k: v for k, v in round_dict.items() if k in summary_df.columns
    }

    # Specify pval columns
    pval_columns = {
        "$p$": 0.05,
        "$p_{corr}$": 0.05,
    }

    # Convert to markdown table
    markdown_table = dataframe_to_markdown(
        summary_df,
        rename_dict={},
        pval_columns=pval_columns,
        round_dict=round_dict,
    )

    return markdown_table
