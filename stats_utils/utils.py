from typing import List, Dict, Optional
import numpy as np
import pandas as pd


def process_summary_table(
    summary_df: pd.DataFrame,
    predictor_rename_dict: Optional[Dict[str, str]] = None,
    exclude_predictors: Optional[List[str]] = None,
    column_rename_dict: Optional[Dict[str, str]] = None,
    round_dict: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Process a summary table DataFrame by renaming columns, applying rounding,
    and filtering predictors.

    Args:
        summary_df (pd.DataFrame): The summary table as a DataFrame.
        predictor_rename_dict (Optional[Dict[str, str]], optional): A
            dictionary to rename the predictors in the summary table. Defaults
            to `None`.
        exclude_predictors (Optional[List[str]], optional): A list of
            predictors to exclude from the summary table. Defaults to `[]`.
        column_rename_dict (Optional[Dict[str, str]], optional): A dictionary
            to rename the summary table columns. Defaults to `None`.
        round_dict (Optional[Dict[str, int]], optional): A dictionary to
            set the rounding precision for each column. Defaults to `None`.

    Returns:
        pd.DataFrame: The processed summary table DataFrame.
    """

    # Rename the columns
    if column_rename_dict is not None:
        summary_df = summary_df.rename(columns=column_rename_dict)

    # Rename the predictors
    if predictor_rename_dict is None:
        summary_df.index = summary_df.index.str.replace("__", " ")
        summary_df.index = summary_df.index.str.replace("_", " ")
        summary_df.index = summary_df.index.str.title()
    else:
        summary_df = summary_df.rename(index=predictor_rename_dict)

    # Drop the intercept row
    summary_df = summary_df.drop(index="Intercept", errors="ignore")

    # Drop any excluded predictors
    if exclude_predictors:
        summary_df = summary_df.drop(index=exclude_predictors, errors="ignore")

    return summary_df


def format_column_repeated_values(
    df: pd.DataFrame, col_name: str
) -> pd.DataFrame:
    """
    Returns a new DataFrame where only the first occurrence of a repeated value
    in the given column is shown, and subsequent ones are left blank.

    Args:
        df (pd.DataFrame): Original DataFrame.
        col_name (str): Name of the column to format.

    Returns:
        pd.DataFrame: New DataFrame with formatted first column.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    formatted_df = df.copy()

    prev_value = None
    for index, value in formatted_df[col_name].items():
        if value == prev_value:
            formatted_df.at[index, col_name] = (
                ""  # Replace repeated values with empty string
            )
        else:
            prev_value = value

    return formatted_df


# Function to apply bold formatting to specified HDI columns where both values
# have the same sign
def bold_hdi_values(row: dict, hdi_lower_col: str, hdi_upper_col: str) -> dict:
    """
    Formats the lower and upper values of a row's HDI (Highest Density
    Interval) with double asterisks if they have the same sign.

    Args:
        row: A dictionary representing a row of data.
        hdi_lower_col: The column name for the lower HDI value.
        hdi_upper_col: The column name for the upper HDI value.

    Returns:
        The modified row dictionary with the HDI values formatted with double
        asterisks if they have the same sign.
    """
    hdi_lower = float(row[hdi_lower_col])
    hdi_upper = float(row[hdi_upper_col])

    # Check if both HDI values have the same sign
    if np.sign(hdi_lower) == np.sign(hdi_upper):
        row[hdi_lower_col] = f"**{row[hdi_lower_col]}**"
        row[hdi_upper_col] = f"**{row[hdi_upper_col]}**"

    return row


def dataframe_to_markdown(
    df: pd.DataFrame,
    round_dict: dict,
    rename_dict: dict,
    pval_columns: Dict[str, float] = None,
    hdi_columns: List[str] = None,
    repeated_value_columns: List[str] = None,
    rename_index: str = "Predictor",
) -> str:
    """
    Processes a pandas DataFrame containing output from some type of
    statistical model by rounding specified columns, renaming columns,
    and converting the DataFrame to a markdown table string.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        round_dict (dict): A dictionary specifying the number of decimal
            places for each column to round to. Example: `{"column1": 2,
            "column2": 3}`
        rename_dict (dict): A dictionary specifying the new column names
            with optional LaTeX formatting. Example: `{"column1":
            "$column_{1}$", "column2": "$column_{2}$"}`
        pval_columns (Dict[str, float]): A dictionary specifying the
            significance level for each p-value column. If specified, the
            column will be converted to a string and significant values will
            be bolded. Example: `{"pval": 0.05, "pval_corr": 0.01}`
        hdi_columns (List[str]): A list of column names representing
            highest density intervals (HDIs) that should be highlighted
            to show "significant" values. Should have two entries
            where the first corresponds to the lower HDI and the second
            corresponds to the upper HDI. Defaults to `[]`.
        repeated_value_columns (List[str]): A list of column names that
            should be formatted to show repeated values. For example, if we
            have multiple target variables and the same predictor variables,
            we can format the target variables to show repeated values.
            Defaults to `[]`.
        rename_index (str): The name to give to the index column. Defaults
            to "Predictor". If `None`, the index is dropped.

    Returns:
        str: A string representing the DataFrame in markdown format.

    Example:
        df = pd.DataFrame(...)
        round_dict = `{"df_resid": 0, "ssr": 2, "ss_diff": 2, "F": 2,
            "Pr(>F)": 3}`
        rename_dict = `{"df_resid": "$df_{R}$", "ssr": "$SS_{R}$",
            "ss_diff": "$SS_{diff}$", "F": "$F$", "Pr(>F)": "$p$"}`
        markdown_str = dataframe_to_latex(df, round_dict, rename_dict,
            'p>|t|')
    """

    # If HDI columns are specified, ensure there are two columns
    if hdi_columns is not None and len(hdi_columns) != 2:
        raise ValueError("hdi_columns must contain two columns")

    # Create a copy of the DataFrame
    df = df.copy()

    # Reset index just in case it's out of order
    df = df.reset_index()

    # Rename the index column
    if rename_index is not None:
        df = df.rename(columns={"index": rename_index})
    else:
        df = df.reset_index(drop=True)
        # Remove any "index" column that might be present
        if "index" in df.columns:
            df = df.drop(columns=["index"])

    # Get rounding precision for each column as a tuple in the column order, as
    # a formatting string
    precisions = tuple([f".{round_dict.get(col, 0)}f" for col in df.columns])

    # Drop any columns that are not in the DataFrame
    round_dict = {
        col: round_dict[col] for col in round_dict if col in df.columns
    }

    # Apply custom formatting based on round_dict
    for col, decimals in round_dict.items():
        # Convert column to numeric if it is not already
        df[col] = pd.to_numeric(df[col])

        # Identify significant rows
        if col in pval_columns:
            significant_rows = df[col].values < pval_columns[col]

        # Round the column to the specified number of decimal places
        df[col] = (
            df[col]
            .apply(lambda x: f"{x:.{decimals}f}" if pd.notnull(x) else "-")
            .astype(str)
        )

        # Bold significant rows and replace 0.000 with "<0.001"
        if col in pval_columns:

            # Replace 0.000 with "<0.001"
            df[col] = df[col].apply(lambda x: "<.001" if x == "0.000" else x)

            # Bold significant rows
            df[col] = df.apply(
                lambda row: (
                    f"**{row[col]}**"
                    if row[col] != "-" and significant_rows[row.name]
                    else row[col]
                ),
                axis=1,
            )

    # Highlight "significant" HDI columns based on both
    # values having the same sign
    if hdi_columns is not None:
        df = df.apply(
            bold_hdi_values,
            axis=1,
            hdi_lower_col=hdi_columns[0],
            hdi_upper_col=hdi_columns[1],
        )

    # Format columns with repeated values
    if repeated_value_columns is not None:
        for col in repeated_value_columns:
            df = format_column_repeated_values(df, col)

    # Rename the columns
    df_renamed = df.rename(columns=rename_dict)

    # Convert to Markdown string
    return df_renamed.to_markdown(index=False, floatfmt=precisions)
