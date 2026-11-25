"""
experiment_tools.py
Helper functions for analyzing randomized experiments.

Usage (put this file in the same folder as your notebook):

    from experiment_tools import balance_by_sample

    # treatment coded 1/0
    balance_by_sample(df, "T", ["age", "male", "region"])

    # treatment coded as text
    balance_by_sample(df, "Treatment", ["LocalService", "Desktop"],
                      treated="treatment", control="control")

    # several samples in one plot
    balance_by_sample({"Eligible": elig, "Ad shown": shown}, "Treatment",
                      ["LocalService", "Desktop"],
                      treated="treatment", control="control")
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def balance_by_sample(samples, treatment_col, covariates, treated=1, control=0,
                      threshold=0.1, title="Covariate balance", name=None):
    """
    Love plot: standardized mean difference (treatment - control) for each covariate.

    samples       : a DataFrame, or a dict {"name": DataFrame} to compare several samples
    treatment_col : column that identifies the groups
    covariates    : list of columns to check (text/category columns become 0/1 dummies;
                    numbers stored as codes, e.g. region 1-5, should be converted first
                    with df["region"] = df["region"].astype("category"))
    treated, control : values of treatment_col for each group (default 1 and 0)
    threshold     : half-width of the "balanced" band (default 0.1)
    title         : plot title
    name          : label for the sample in the returned table (single DataFrame only)

    Returns a DataFrame with the SMD of every covariate in every sample.
    """
    # 0. Accept a single DataFrame as well as a dict of samples
    if isinstance(samples, pd.DataFrame):
        samples = {name or title: samples}

    # 1. Standardized mean difference (SMD) for every covariate, in every sample
    rows = []
    for sample_name, data in samples.items():
        X = pd.get_dummies(data[covariates], drop_first=True).astype(float)
        t1 = X[data[treatment_col] == treated]
        t0 = X[data[treatment_col] == control]
        if len(t1) == 0 or len(t0) == 0:
            raise ValueError(
                f"No rows for treated={treated!r} or control={control!r} in sample "
                f"'{sample_name}'. Values in '{treatment_col}': "
                f"{data[treatment_col].unique().tolist()}"
            )
        for col in X.columns:
            pooled_sd = np.sqrt((t1[col].var() + t0[col].var()) / 2)
            if pooled_sd == 0:          # variable is constant in this sample: nothing to compare
                continue
            smd = (t1[col].mean() - t0[col].mean()) / pooled_sd
            rows.append({"Sample": sample_name, "Variable": col, "SMD": smd})
    res = pd.DataFrame(rows)

    # 2. Plot: one row per covariate, one marker per sample
    fig, ax = plt.subplots(figsize=(8, 1 + 0.8 * res["Variable"].nunique()))
    variables = list(res["Variable"].unique())
    markers = ["o", "s", "^", "D", "v"]
    n = res["Sample"].nunique()
    for k, (sample_name, part) in enumerate(res.groupby("Sample", sort=False)):
        offset = (k - (n - 1) / 2) * 0.15        # small vertical shift so markers don't overlap
        y = [variables.index(v) + offset for v in part["Variable"]]
        ax.scatter(part["SMD"], y, s=70, marker=markers[k % len(markers)],
                   label=sample_name, zorder=3)
    ax.axvline(0, color="black", lw=1)
    ax.axvspan(-threshold, threshold, color="lightgray", alpha=0.3,
               label=f"±{threshold} (balanced)")
    ax.set_yticks(range(len(variables)), variables)
    limit = max(0.3, res["SMD"].abs().max() * 1.2)   # widen the axis if an SMD is large
    ax.set_xlim(-limit, limit)
    ax.set_title(title)
    if n > 1:                                   # several samples: legend identifies them
        ax.set_xlabel("Standardized mean difference (treatment − control)")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    else:                                       # one sample: no legend, explain the band in the label
        ax.set_xlabel(f"Standardized mean difference (treatment − control); "
                      f"gray band = ±{threshold}")
    plt.tight_layout()
    plt.show()
    return res
