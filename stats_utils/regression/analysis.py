from dataclasses import dataclass
from types import MethodType
from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.utils import resample
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats.anova import anova_lm


def add_bootstrap_methods_to_ols(
    results: RegressionResults,
) -> RegressionResults:
    """
    Add bootstrap methods to the OLS results class.

    Args:
        results (RegressionResults): The results of an OLS regression.

    Returns:
        RegressionResults: The results object with the bootstrap methods added.

    Example:
        ```
        # Assuming `results` is the output of an OLS regression
        results = add_bootstrap_methods_to_ols(results)
        results.bootstrap(n_bootstraps=2000)
        conf_int = results.conf_int_bootstrap()

        # Access the pvals
        pvals = results.pvalues_bootstrap
        ```
    """

    def bootstrap(
        self, n_bootstraps: int = 2000, random_state: int = 42
    ) -> None:
        """
        Perform a bootstrap on the OLS model, allowing for the estimation of
        confidence intervals.

        Results are stored in the `coefs_bootstrap_samples` attribute.

        Args:
            n_bootstraps (int, optional): Number of bootstrap samples.
                Defaults to `1000`.
            random_state (int, optional): Random state for reproducibility.
                Defaults to `42`.
        """

        # Get the exog and endog variables
        X = self.model.exog
        y = self.model.endog

        # Create randomstate
        rng = np.random.RandomState(42)

        # Create a list to store the bootstrapped coefficients
        coef_samples = []

        # Run the bootstrap, looping over the number of bootstraps
        for _ in range(n_bootstraps):

            # Resample the data with replacement
            x_resampled, y_resampled = resample(X, y, random_state=rng)

            # Fit the model to the resampled data
            model_resampled = sm.OLS(y_resampled, x_resampled)

            # Fit the resampled model and store the coefficients
            results_resampled = model_resampled.fit()

            coef_samples.append(results_resampled.params)

        self.coefs_bootstrap_samples = pd.DataFrame(coef_samples)

    def conf_int_bootstrap(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Get the confidence intervals (and p values) for the coefficients of the
        OLS model, based on the bootstrapped coefficients.

        Useful for situations where the assumptions of the OLS model are not
        met.

        Args:
            alpha (float, optional): Alpha level. Defaults to `0.05`.

        Returns:
            pd.DataFrame: Dataframe of confidence intervals, with
                columns `0` and `1`
        """

        # Get the lower and upper bounds of the confidence interval
        lower_bound = self.coefs_bootstrap_samples.quantile(alpha / 2)
        upper_bound = self.coefs_bootstrap_samples.quantile(1 - alpha / 2)

        # Get the p-values
        p_values = (self.coefs_bootstrap_samples < 0).sum(
            axis=0
        ) / self.coefs_bootstrap_samples.shape[0]
        p_values = 2 * np.minimum(p_values, 1 - p_values)

        # Store the confidence intervals in the same format as the normal
        # conf_int method
        conf_int = pd.DataFrame(
            np.array([lower_bound, upper_bound]).T,
            columns=[0, 1],
            index=self.params.index,
        )

        # Store the p values in the same format as the normal pvalues method
        p_values.index = self.params.index
        self.pvalues_bootstrap = p_values

        return conf_int

    results.bootstrap = MethodType(bootstrap, results)
    results.conf_int_bootstrap = MethodType(conf_int_bootstrap, results)

    return results


@dataclass
class ModelOutput:
    """
    Dataclass to store the output of the sequential regression function

    Attributes:
        models (List[smf.ols]): List of fitted models.
        anova_results (pd.DataFrame): ANOVA results.
        r2s (List[float]): List of adjusted r2s.
        summaries (List[str]): List of model summaries.
        n_solutions (int): Number of solutions.
        y_var (str): Name of dependent variable.
    """

    models: List[smf.ols]
    anova_results: pd.DataFrame
    r2s: List[float]
    summaries: List[str]
    n_solutions: int
    y_var: str


def sequential_regression(
    data: pd.DataFrame,
    y: str,
    n_solutions: int = 4,
    covariates: List[str] = [],
    n_bootstraps: int = 2000,
) -> Tuple[List[smf.ols], pd.DataFrame, List[float]]:
    """
    Fits a series of regression models across different factor solutions.

    Args:
        data (pd.DataFrame): Dataframe containing dependent variable,
            covariates (age and gender), and factor scores. Assumes that
            factor scores are named `Sol{N}_ML{M}` where `N` is the total
            number of factors and `M` is the number of each factor within
            that solution.
        y (str): Name of dependent variable.
        n_solutions (int, optional): Number of solutions. Defaults to `4`.
        covariates (List[str]): List of covariates to include in the model
            (in addition to age and gender). Defaults to `[]`.
        n_bootstraps (int, optional): Number of bootstraps to run. Defaults
            to `2000`.

    Returns:
        Tuple[List[smf.ols], pd.DataFrame, List[float]]: Returns the list of
            fitted models, the ANOVA table, and a list of adjusted r2s.
    """

    # List to store model fits
    models = []

    # Loop over number of solutions
    for n_factors in range(n_solutions + 1):
        # Get predictors
        predictors = " + ".join(
            ["Sol{0}_ML{1}".format(n_factors, i + 1) for i in range(n_factors)]
        )
        if len(predictors) > 0:
            predictors = " + " + predictors
        covariates_string = " + ".join(covariates)
        if len(covariates) > 0:
            covariates_string = " + " + covariates_string
        predictors = "age + gender" + covariates_string + predictors

        # Specify model
        formula_string = "{0} ~ {1}".format(y, predictors)
        model = smf.ols(formula_string, data=data)

        # Fit model
        fitted_model = model.fit()

        # Replace the class with the bootstrap results class
        fitted_model = add_bootstrap_methods_to_ols(fitted_model)

        # Run bootstrap
        fitted_model.bootstrap(n_bootstraps)

        # Add to list
        models.append(fitted_model)

    # Run ANOVA on fits
    anova_results = anova_lm(*models)

    # get adjusted r2s for each model and put into a dataframe
    r2s = []

    for m in models:
        r2s.append(m.rsquared_adj)

    # Get summaries
    summaries = [m.summary() for m in models]

    return ModelOutput(models, anova_results, r2s, summaries, n_solutions, y)
