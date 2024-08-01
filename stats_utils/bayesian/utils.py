from typing import Dict, List, Optional

import arviz as az

from ..utils import dataframe_to_markdown, process_summary_table


def arviz_to_markdown_table(
    az_result: az.InferenceData,
    predictor_rename_dict: Optional[Dict[str, str]] = None,
    exclude_predictors: Optional[List[str]] = [],
    column_rename_dict: Optional[Dict[str, str]] = None,
    round_dict: Optional[Dict[str, int]] = None,
    credible_interval: float = 0.95,
) -> str:
    """
    Convert the summary table of a Bayesian regression result from ArviZ to
    a markdown table.

    This function takes an ArviZ `InferenceData` object, typically containing
    the results of a Bayesian regression model, and converts the summary
    statistics into a markdown table. The table can be customized by renaming
    predictors, columns, and rounding values. The resulting table is formatted
    with LaTeX-compatible symbols for use in documents.

    Args:
        az_result (az.InferenceData):
            The ArviZ `InferenceData` object containing the results of a
            Bayesian regression analysis.
        predictor_rename_dict (Optional[Dict[str, str]], optional):
            A dictionary to rename the predictors in the summary table. Keys
            are the original predictor names, and values are the desired new
            names. If not provided, the predictor names will remain unchanged.
            Defaults to `None`.
        exclude_predictors (Optional[List[str]], optional):
            A list of predictors to exclude from the summary table. These
            predictors will be removed from the table entirely. Defaults to an
            empty list (`[]`).
        column_rename_dict (Optional[Dict[str, str]], optional):
            A dictionary to rename the columns in the summary table. Keys are
            the original column names (e.g., "mean", "sd"), and values are the
            desired new names (e.g., "$\\mu$", "$\\sigma$"). Defaults to a
            pre-specified dictionary that renames common statistical terms to
            LaTeX symbols.
        round_dict (Optional[Dict[str, int]], optional):
            A dictionary specifying the rounding precision for each column.
            Keys are the column names (e.g., "$\\mu$", "$\\sigma$"), and values
            are the number of decimal places to round to. Defaults to a
            pre-specified dictionary with reasonable rounding for typical
            Bayesian output.
        credible_interval (float, optional):
            The credible interval range to be used, expressed as a proportion
            (e.g., 0.95 for a 95% credible interval). This will determine the
            bounds used in the summary table. Defaults to `0.95`.

    Returns:
        str:
            A markdown-formatted string representing the summary table of the
            Bayesian regression results.

    Example:
        ```python
        import arviz as az

        # Assuming `az_result` is an ArviZ InferenceData object from a
        # Bayesian model
        markdown_table = bayesian_to_markdown_table(
            az_result,
            predictor_rename_dict={"x1": "Variable 1", "x2": "Variable 2"},
            exclude_predictors=["Intercept"],
            credible_interval=0.90
        )
        print(markdown_table)
        ```

    Notes:
        - The function uses the `arviz.summary()` method to extract summary
          statistics from the `InferenceData` object.
        - If the `credible_interval` is set to a value other than 0.95, the
          column names for the lower and upper credible intervals will be
          adjusted accordingly.
        - The function assumes that a `dataframe_to_markdown` utility is
          available to convert the processed DataFrame into a markdown table.
    """

    # Default column renaming dict
    if column_rename_dict is None:

        # Get the lower and upper bounds for the credible interval
        lower_bound = (1 - credible_interval) / 2 * 100
        upper_bound = (1 + credible_interval) / 2 * 100

        column_rename_dict = {
            "mean": "$\\mu$",
            "sd": "$\\sigma$",
            f"hdi_{lower_bound:.1f}%": f"$HDI_{{{lower_bound:.1f}}}$",
            f"hdi_{upper_bound:.1f}%": f"$HDI_{{{upper_bound:.1f}}}$",
        }

    # Default rounding precision dict
    if round_dict is None:
        round_dict = {
            "$\\mu$": 2,
            "$\\sigma$": 2,
            "$CI_{lower}$": 2,
            "$CI_{upper}$": 2,
        }

    # Convert the summary to a DataFrame
    summary_df = az.summary(az_result, hdi_prob=credible_interval)

    # Drop diagnostic columns
    summary_df = summary_df.drop(
        columns=[
            "r_hat",
            "ess_bulk",
            "ess_tail",
            "mcse_mean",
            "mcse_sd",
        ]
    )

    # Drop any rows with values of the index that contain "_sigma"
    summary_df = summary_df[~summary_df.index.str.contains("_sigma")]

    # Process the summary table
    summary_df = process_summary_table(
        summary_df,
        predictor_rename_dict=predictor_rename_dict,
        exclude_predictors=exclude_predictors,
        column_rename_dict=column_rename_dict,
        round_dict=round_dict,
    )

    # Convert to markdown table
    markdown_table = dataframe_to_markdown(
        summary_df,
        rename_dict={},
        pval_columns=[],
        hdi_columns=[
            f"$HDI_{{{lower_bound:.1f}}}$",
            f"$HDI_{{{upper_bound:.1f}}}$",
        ],
        round_dict=round_dict,
    )

    return markdown_table
