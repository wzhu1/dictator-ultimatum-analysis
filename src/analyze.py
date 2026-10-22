"""Analyze the simulated experiment: balance, treatment effects, mediation-style
regressions, ultimatum acceptance, and a simulation-based power analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from simulate import TRUE, simulate

ROOT = Path(__file__).resolve().parents[1]


def balance_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["female", "econ_major"]:
        a, b = df.loc[df.device == 0, col], df.loc[df.device == 1, col]
        rows.append({"covariate": col, "control": a.mean(), "device": b.mean(),
                     "p_value": stats.ttest_ind(a, b).pvalue})
    return pd.DataFrame(rows)


def fit_models(df: pd.DataFrame) -> dict:
    hc = {"cov_type": "HC1"}
    return {
        "give": smf.ols("dictator_give ~ device + female + econ_major", df).fit(**hc),
        "guilt": smf.ols("guilt ~ device + dictator_give + female + econ_major", df).fit(**hc),
        "happiness": smf.ols("happiness ~ device + I(10 - dictator_give) + accepted", df).fit(**hc),
        "accept": smf.logit("accepted ~ ultimatum_offer", df).fit(disp=False),
    }


def coef_table(models: dict) -> pd.DataFrame:
    spec = [
        ("give", "device", TRUE["device_effect_on_dictator_give"]),
        ("guilt", "device", TRUE["device_effect_on_guilt"]),
        ("guilt", "dictator_give", TRUE["give_effect_on_guilt"]),
        ("happiness", "device", TRUE["device_effect_on_happiness"]),
        ("happiness", "I(10 - dictator_give)", TRUE["keep_effect_on_happiness"]),
    ]
    rows = []
    for model, term, truth in spec:
        m = models[model]
        lo, hi = m.conf_int().loc[term]
        rows.append({"outcome": model, "term": term, "estimate": m.params[term], "ci_low": lo,
                     "ci_high": hi, "p_value": m.pvalues[term], "true_value": truth,
                     "covers_truth": lo <= truth <= hi})
    return pd.DataFrame(rows)


def power_curve(sizes=(100, 200, 300, 400, 600), reps: int = 200) -> pd.DataFrame:
    """Share of simulated experiments where the device effect on giving is significant at 5%."""
    rows = []
    for n in sizes:
        hits = 0
        for r in range(reps):
            d = simulate(n, seed=10_000 + r)
            p = smf.ols("dictator_give ~ device", d).fit(cov_type="HC1").pvalues["device"]
            hits += p < 0.05
        rows.append({"n": n, "power": hits / reps})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, power: pd.DataFrame, accept_model) -> None:
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for t, g in df.groupby("treatment"):
        axes[0].hist(g.dictator_give, bins=np.arange(0, 10.5, 0.5), alpha=0.6, label=t)
    axes[0].set(title="Dictator giving by arm", xlabel="dollars given (of 10)", ylabel="participants")
    axes[0].legend()

    means = df.groupby("treatment")[["guilt", "happiness"]].agg(["mean", "sem"])
    x = np.arange(2)
    for i, t in enumerate(["control", "device"]):
        axes[1].bar(x + i * 0.35, [means.loc[t, ("guilt", "mean")], means.loc[t, ("happiness", "mean")]], 0.35,
                    yerr=[1.96 * means.loc[t, ("guilt", "sem")], 1.96 * means.loc[t, ("happiness", "sem")]],
                    label=t, capsize=4)
    axes[1].set_xticks(x + 0.175, ["guilt", "happiness"])
    axes[1].set(title="Self-reported emotions (95% CI)", ylabel="1-7 scale")
    axes[1].legend()

    grid = pd.DataFrame({"ultimatum_offer": np.linspace(0, 6, 100)})
    axes[2].plot(grid.ultimatum_offer, accept_model.predict(grid))
    jitter = df.accepted + np.random.default_rng(0).normal(0, 0.02, len(df))
    axes[2].scatter(df.ultimatum_offer, jitter, s=6, alpha=0.3)
    axes[2].set(title="Ultimatum acceptance (logit fit)", xlabel="dollars offered (of 10)", ylabel="P(accept)")
    fig.tight_layout()
    fig.savefig(out / "experiment_overview.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(power.n, power.power, marker="o")
    ax.axhline(0.8, ls="--", color="gray")
    ax.set(title="Power to detect the device effect on giving", xlabel="total sample size", ylabel="power", ylim=(0, 1.05))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "power_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "experiment.csv")
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    bal = balance_table(df)
    models = fit_models(df)
    coefs = coef_table(models)
    power = power_curve()
    plot(df, power, models["accept"])

    bal.to_csv(reports / "balance.csv", index=False)
    coefs.to_csv(reports / "estimates_vs_truth.csv", index=False)
    power.to_csv(reports / "power.csv", index=False)
    (reports / "regressions.txt").write_text("\n\n".join(str(m.summary()) for m in models.values()))

    print("Balance\n", bal.round(3).to_string(index=False))
    print("\nEstimates vs. ground truth\n", coefs.round(3).to_string(index=False))
    print("\nPower\n", power.to_string(index=False))


if __name__ == "__main__":
    main()
