from typing import Dict, Optional

import pandas as pd

from ..utils import dataframe_to_markdown, process_summary_table


def icc_to_markdown_table(
    icc_results: pd.DataFrame,
    variable_rename_dict: Optional[Dict[str, str]] = None,
    column_rename_dict: Optional[Dict[str, str]] = None,
    round_dict: Optional[Dict[str, int]] = None,
) -> str:
    """
    Convert the summary table produced by the
    `calculate_intraclass_correlations` function to a markdown table.

    Args:
        icc_results (pd.DataFrame): The DataFrame containing the intraclass
            correlation results.
        variable_rename_dict (Optional[Dict[str, str]], optional): A
            dictionary to rename variables table. If not included, predictors
            will be tidied slightly instead. Defaults to `None`.
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
        str: The markdown table representing the summary table of the ICC
             results.

    """

    # Default column renaming dict
    if column_rename_dict is None:
        column_rename_dict = {
            "ICC": "$ICC$",
            "F": "$F$",
            "df1": "$df_1$",
            "df2": "$df_2$",
            "pval": "$p$",
            "CI2.5%": "$CI_{2.5}$",
            "CI97.5%": "$CI_{97.5}$",
        }

    # Default rounding precision dict
    if round_dict is None:
        round_dict = {
            "$ICC$": 2,
            "$F$": 2,
            "$df_1$": 0,
            "$df_2$": 0,
            "$p$": 3,
            "$CI_{2.5}$": 2,
            "$CI_{97.5}$": 2,
        }

    # Drop the 'Type' and 'Description' columns
    summary_df = icc_results.drop(columns=["Type", "Description"])

    # Split the 'CI95%' column into 'CI2.5%' and 'CI97.5%' columns
    summary_df["CI2.5%"] = summary_df["CI95%"].apply(lambda x: x[0])
    summary_df["CI97.5%"] = summary_df["CI95%"].apply(lambda x: x[1])

    # Convert the CI columns to float
    summary_df["CI2.5%"] = summary_df["CI2.5%"].astype(float)
    summary_df["CI97.5%"] = summary_df["CI97.5%"].astype(float)

    # Drop the original 'CI95%' column
    summary_df = summary_df.drop(columns=["CI95%"])

    # Set 'variable' as the single index
    summary_df = summary_df.set_index("variable")
    summary_df.index.name = None

    # Process the summary table
    summary_df = process_summary_table(
        summary_df,
        predictor_rename_dict=variable_rename_dict,
        exclude_predictors=None,
        column_rename_dict=column_rename_dict,
        round_dict=round_dict,
    )

    # Convert to markdown table
    markdown_table = dataframe_to_markdown(
        summary_df,
        rename_dict={},
        pval_columns=None,
        round_dict=round_dict,
        rename_index='Variable'
    )

    return markdown_table
