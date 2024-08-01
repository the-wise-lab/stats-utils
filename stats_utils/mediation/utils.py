import pandas as pd
from typing import Dict, Optional
from ..utils import dataframe_to_markdown, process_summary_table
import re


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

    # Get the summary df
    summary_df = mediation_result.copy()

    # Replace the index with the "path" column values
    # for compatibility with general formatting function
    summary_df.index = summary_df["path"]
    # and drop the "path" column
    summary_df = summary_df.drop(columns="path")

    # Process the summary table using the common function
    summary_df = process_summary_table(
        summary_df=summary_df,
        predictor_rename_dict=variable_rename_dict,
        exclude_predictors=[],
        column_rename_dict=column_rename_dict,
        round_dict=round_dict,
    )

    # Move current index values to a new column called "Path"
    summary_df = summary_df.reset_index()
    summary_df = summary_df.rename(columns={"path": "Path"})

    # Put variable names for indirect paths in brackets
    summary_df["Path"] = summary_df["Path"].apply(
        lambda x: re.sub(r"Indirect (.+)", r"Indirect (\1)", x)
    )

    # Drop the "sig" column if it exists
    if "sig" in summary_df.columns:
        summary_df = summary_df.drop(columns="sig")

    # Convert to markdown table
    markdown_table = dataframe_to_markdown(
        summary_df,
        rename_dict={},
        pval_columns={"$p$": 0.05},
        round_dict=round_dict,
        rename_index=None,
    )

    return markdown_table
