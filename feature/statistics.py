from typing import Dict, Any
import numpy as np
from scipy.stats import ttest_ind, f_oneway
from sklearn.linear_model import LinearRegression
from scipy.stats import t
from .base import StatisticalTest
import statsmodels.api as sm
from lfp_analysis.registery import register

@register("statistics","t-test")
class TTest(StatisticalTest):
    def compare(self, data: Dict[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        # Require exactly two groups
        if len(data) != 2:
            raise ValueError("TTest requires exactly 2 groups.")
        groups = list(data.values())
        # Use kwargs for ttest_ind options like equal_var
        stat, pval = ttest_ind(groups[0], groups[1], **kwargs)
        return {
            "test": "t-test",
            "statistic": stat,
            "p_value": pval,
            "n_samples": [len(g) for g in groups],
        }


@register("statistics","anova")
class ANOVA(StatisticalTest):
    def compare(self, data: Dict[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        # Need at least 2 groups
        if len(data) < 2:
            raise ValueError("ANOVA requires at least 2 groups.")
        groups = list(data.values())
        stat, pval = f_oneway(*groups)
        return {
            "test": "anova",
            "statistic": stat,
            "p_value": pval,
            "n_groups": len(groups),
            "n_samples": [len(g) for g in groups],
        }

@register("statistics","regression_significance")
class RegressionSignificance(StatisticalTest):
    def compare(self, data: Dict[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        # Require X and y keys
        if "X" not in data or "y" not in data:
            raise ValueError("RegressionSignificance requires 'X' and 'y' keys.")

        X = data["X"]
        y = data["y"]

        n_samples, n_features = X.shape

        # Fit linear regression (no intercept by default, so add if needed)
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        coef = model.coef_
        intercept = model.intercept_

        # Predictions and residuals
        y_pred = model.predict(X)
        residuals = y - y_pred

        # Degrees of freedom
        dof = n_samples - n_features - 1

        # Residual variance estimate
        residual_var = np.sum(residuals**2) / dof

        # Add intercept column to X for covariance calculation
        X_design = np.column_stack((np.ones(n_samples), X))

        # Compute covariance matrix of coefficients: sigma^2 * (X'X)^-1
        XtX_inv = np.linalg.inv(X_design.T @ X_design)
        coef_var = residual_var * np.diag(XtX_inv)

        # Standard errors
        std_err = np.sqrt(coef_var)

        # Coefficients with intercept included
        coefs_with_intercept = np.hstack([intercept, coef])

        # t statistics
        t_stats = coefs_with_intercept / std_err

        # Two-sided p-values
        p_values = 2 * (1 - t.cdf(np.abs(t_stats), dof))

        return {
            "test": "regression_significance",
            "coefficients": coefs_with_intercept,
            "std_errors": std_err,
            "t_values": t_stats,
            "p_values": p_values,
            "degrees_of_freedom": dof,
            "residual_variance": residual_var,
        }

@register("statistics","regression_significance1")
class RegressionSignificance1(StatisticalTest):
    def compare(self, data: Dict[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        # Expect keys 'X' and 'y'
        if "X" not in data or "y" not in data:
            raise ValueError("RegressionSignificance requires 'X' and 'y' keys.")
        
        X = data["X"]
        y = data["y"]

        # Add intercept if not already present
        if not np.allclose(X[:, 0], 1):
            X = sm.add_constant(X)

        model = sm.OLS(y, X)
        results = model.fit()

        # Return coefficients, standard errors, t-values and p-values in dict
        coefs = results.params
        std_err = results.bse
        t_vals = results.tvalues
        p_vals = results.pvalues

        return {
            "test": "regression_significance",
            "coefficients": coefs,
            "std_errors": std_err,
            "t_values": t_vals,
            "p_values": p_vals,
            "rsquared": results.rsquared,
            "adj_rsquared": results.rsquared_adj,
        }