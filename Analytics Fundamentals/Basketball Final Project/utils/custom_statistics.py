import numpy as np
from scipy.stats.stats import pearsonr


def get_outliers_info(df, nstd):
    outliers = {}
    outliers_index = []
    for col in df.columns:
        values = df[col]
        upper_bound = values.mean() + nstd * values.std()
        lower_bound = values.mean() - nstd * values.std()
        lower_outliers = df[df[col] < lower_bound][col]
        upper_outliers = df[df[col] > upper_bound][col]
        outliers_index.extend(lower_outliers.index)
        outliers_index.extend(upper_outliers.index)

        if(len(lower_outliers) > 0):
            outliers[col] = {
                "lower_bound": lower_bound,
                "nlower_outliers": len(lower_outliers),
                "lower_outliers": lower_outliers
            }

        if(len(upper_outliers) > 0 and len(lower_outliers) > 0):
            outliers[col].update(
                {
                    "upper_bound": upper_bound,
                    "nupper_outliers": len(upper_outliers),
                    "upper_outliers": upper_outliers,
                }
            )
        elif(len(upper_outliers) > 0):
            outliers[col] = {
                "upper_bound": upper_bound,
                "nupper_outliers": len(upper_outliers),
                "upper_outliers": upper_outliers,
            }

    unique_outliers = np.unique(outliers_index)
    outliers.update({
        "nunique_outliers": len(unique_outliers),
        "unique_outliers": unique_outliers
    })

    return outliers


def personr_analysis(df, y, significance_level):
    personr_analysis = {}
    for column in df.columns:
        corr, p = pearsonr(x=df[column], y=y)
        correlation_force = "no relationship"
        if(corr > 0.75):
            correlation_force = "strong"
        elif(corr > 0.5):
            correlation_force = "moderate"
        elif(corr > 0.25):
            correlation_force = "weak"

        personr_analysis[column] = {
            "correlation": corr,
            "p": p,
            "is_significant": p < significance_level,
            "correlation_type": "Negative" if corr < 0 else "Positive",
            "correlation_force": correlation_force
        }
    return personr_analysis
