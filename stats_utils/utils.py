from typing import List
import pandas as pd


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


def dataframe_to_markdown(
    df: pd.DataFrame,
    round_dict: dict,
    rename_dict: dict,
    pval_column: str = None,
    repeated_value_columns: List[str] = None,
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
        pval_column (str): The name of the column containing p-values. If
            specified, the column will be converted to a string and
            significant values will be bolded. Defaults to `None`.
        repeated_value_columns (List[str]): A list of column names that
            should be formatted to show repeated values. For example, if we
            have multiple target variables and the same predictor variables,
            we can format the target variables to show repeated values.
            Defaults to `[]`.

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

    # Create a copy of the DataFrame
    df = df.copy()

    # Reset index just in case it's out of order
    df = df.reset_index()

    # Rename the index column as "predictor"
    df = df.rename(columns={"index": "Predictor"})

    # Get rounding precision for each column as a tuple in the column order, as
    # a formatting string
    precisions = tuple([f".{round_dict.get(col, 0)}f" for col in df.columns])

    # Apply custom formatting based on round_dict
    for col, decimals in round_dict.items():
        # Convert column to numeric if it is not already
        df[col] = pd.to_numeric(df[col])

        # Identify significant rows
        if col == pval_column:
            significant_rows = df[col].values < 0.05

        # Round the column to the specified number of decimal places
        df[col] = (
            df[col]
            .apply(lambda x: f"{x:.{decimals}f}" if pd.notnull(x) else "-")
            .astype(str)
        )

        # Bold significant rows
        if col == pval_column:
            df[col] = df.apply(
                lambda row: (
                    f"**{row[col]}**"
                    if row[col] != "-" and significant_rows[row.name]
                    else row[col]
                ),
                axis=1,
            )

    # Format columns with repeated values
    if repeated_value_columns is not None:
        for col in repeated_value_columns:
            df = format_column_repeated_values(df, col)

    # Rename the columns
    df_renamed = df.rename(columns=rename_dict)

    # Convert to Markdown string
    return df_renamed.to_markdown(index=False, floatfmt=precisions)
