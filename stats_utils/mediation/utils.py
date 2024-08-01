import pandas as pd
from typing import Dict, Optional
from ..utils import dataframe_to_markdown


def mediation_analysis_to_markdown_table(
    mediation_result: pd.DataFrame,
    variable_rename_dict: Optional[Dict[str, str]] = None,
    column_rename_dict: Optional[Dict[str, str]] = None,
    round_dict: Optional[Dict[str, int]] = None,
) -> str:
    """
    Convert the summary table of a [`pingouin`](https://pingouin-stats.org/)
    causal mediation analysis result to a markdown table.

    Args:
        mediation_result (sm.RegressionResultsWrapper):
            The results of a `pingouin` mediation analysis.
        variable_rename_dict (Optional[Dict[str, str]], optional): A
            dictionary to rename the variables in the summary table. If not
            included, predictors will be tidied slightly instead. Defaults to
            `None`.
        column_rename_dict (Optional[Dict[str, str]], optional): A
            dictionary to rename the summary table columns. Defaults to a
            pre-specified dictionary if not provided.
        round_dict (Optional[Dict[str, int]], optional): A dictionary to set
            the rounding precision for each column. Defaults to a pre-specified
            dictionary if not provided.

    Returns:
        str: The markdown table representing the summary table of the mediation
            analysis result

    Example:

    """

    # Default column renaming dict
    if column_rename_dict is None:
        column_rename_dict = {
            "path": "Path",
            "coef": "$\\beta$",
            "se": "$\\beta_{SE}$",
            "pval": "$p$",
            "CI[2.5%]": "$CI_{2.5}$",
            "CI[97.5%]": "$CI_{97.5}$",
        }

    # Default rounding precision dict
    if round_dict is None:
        round_dict = {
            "$\\beta$": 2,
            "$\\beta_{SE}$": 2,
            "$p$": 3,
            "$CI_{2.5}$": 2,
            "$CI_{97.5}$": 2,
        }

    # Convert the summary table to a DataFrame
    summary_df = mediation_result.copy()

    # Rename the columns
    summary_df = summary_df.rename(columns=column_rename_dict)

    # Rename the predictors
    # If we don't have a rename dict, tidy the predictor names slightly
    if variable_rename_dict is None:
        # Replace __ with space within rows of the "Path" column
        summary_df["Path"] = summary_df["Path"].str.replace("__", " ")
        # Replace _ with space
        summary_df["Path"] = summary_df["Path"].str.replace("_", " ")
        # Capitalize the first letter of each word
        summary_df["Path"] = summary_df["Path"].str.title()
    else:
        # Otherwise, replace values according to the specified dictionary
        summary_df["Path"] = summary_df["Path"].map(variable_rename_dict)

    # Drop the "sig" column
    summary_df = summary_df.drop(columns="sig")

    # Remove columns that arne't needed from the rounding dict
    round_dict = {
        k: v for k, v in round_dict.items() if k in summary_df.columns
    }

    # Specify pval columns
    pval_columns = {
        "$p$": 0.05,
    }

    # Convert to markdown table
    markdown_table = dataframe_to_markdown(
        summary_df,
        rename_dict={},
        pval_columns=pval_columns,
        round_dict=round_dict,
        rename_index=None
    )

    return markdown_table
