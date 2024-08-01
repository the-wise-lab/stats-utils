from typing import List

import pandas as pd
from pingouin import intraclass_corr


def calculate_intraclass_correlations(
    data: pd.DataFrame,
    variables: List[str],
    variable_col: str = "variable",
    subject_col: str = "subjectID",
    timepoint_col: str = "timepoint",
    value_col: str = "value",
    icc_type: str = "ICC3",
) -> pd.DataFrame:
    """
    Computes intraclass correlation coefficients (ICCs) for a specified list of
    variables across multiple subjects and timepoints (or raters). This
    function is designed for use with long-format DataFrames, where each row
    represents a single observation, including the subject ID, timepoint,
    variable name, and corresponding value. It supports various types of ICC
    calculations, as implemented in the corresponding function from
    `pingouin`. The output is a DataFrame containing the ICC
    values, associated statistics, and confidence intervals for each variable.

    Args:
        data (pd.DataFrame): The DataFrame containing the data variables.
            This should be a long-format DataFrame with one row per
            observation.
        variables (List[str]): List of variables (as strings) for which
            to calculate intraclass correlations.
        variable_col (str): The name of the column containing the variable
            names. Defaults to `"variable"`.
        subject_col (str): The name of the column containing the subject IDs.
            Defaults to `"subjectID"`.
        timepoint_col (str): The name of the column containing the timepoint
            (rater). Defaults to `"timepoint"`.
        value_col (str): The name of the column containing the values to be
            rated. Defaults to `"value"`.
        icc_type (str): The type of intraclass correlation. Defaults to
            `"ICC3"`.

    Returns:
        pd.DataFrame: A DataFrame containing intraclass correlation results for
        each variable.

    Example:
        ```python
        import pandas as pd
        import numpy as np

        # Number of subjects
        num_subjects = 5

        # Initial values for timepoint 1
        values_timepoint_1 = {
            'A': np.random.randint(5, 15, num_subjects),
            'B': np.random.randint(5, 15, num_subjects)
        }

        # Adding Gaussian noise to create values for timepoint 2
        noise = np.random.normal(0, 0.5, num_subjects)

        # Constructing the DataFrame
        data = pd.DataFrame({
            'subjectID': np.repeat(np.arange(1, num_subjects + 1), 4),
            'timepoint': [1, 1, 2, 2] * num_subjects,
            'variable': ['A', 'B', 'A', 'B'] * num_subjects,
            'value': np.concatenate([
                values_timepoint_1['A'],
                values_timepoint_1['B'],
                values_timepoint_1['A'] + noise,
                values_timepoint_1['B'] + noise
            ])
        })

        # Calculate intraclass correlations for variables A, B, and C
        icc_results = calculate_intraclass_correlations(
            data=data,
            variables=['A', 'B'],
            variable_col='variable',
            subject_col='subjectID',
            timepoint_col='timepoint',
            value_col='value',
            icc_type='ICC3'
        )

        print(icc_results)
        ```
    """
    # Initialize a list to store the intraclass correlation results for each
    # variable
    iccs = []

    # Loop through each variable to compute its intraclass correlation
    for variable in variables:
        # Filter the data for the current variable
        variable_data = data[data[variable_col] == variable]

        # Calculate intraclass correlation
        icc_res = intraclass_corr(
            data=variable_data,
            targets=subject_col,
            raters=timepoint_col,
            ratings=value_col,
        )

        # Add a column to the result indicating the current variable
        icc_res["variable"] = variable

        # Append the result to the list of intraclass correlations
        iccs.append(icc_res)

    # Concatenate the individual results into a single DataFrame
    iccs = pd.concat(iccs)

    # Filter to only include results of the desired type
    iccs = iccs[iccs["Type"] == icc_type]

    return iccs
