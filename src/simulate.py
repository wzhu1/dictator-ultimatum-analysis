"""Simulate an online Dictator / Ultimatum experiment with a decision-device treatment.

Design (between subjects, 2 arms):
  * control  - participants choose their own allocation of a $10 endowment
  * device   - participants may delegate the choice to an external decision device
               (a randomizer), which is thought to dampen feelings of responsibility

Every participant plays one Dictator round and one Ultimatum round (as proposer),
then reports guilt and happiness on 1-7 scales.

The data are synthetic, generated from known "true" parameters so the analysis in
analyze.py can be checked against ground truth. No real participant data is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ENDOWMENT = 10.0

# Ground-truth effects the analysis should recover.
TRUE = {
    "device_effect_on_dictator_give": -0.9,   # dollars
    "device_effect_on_guilt": -0.6,           # scale points, holding giving fixed
    "give_effect_on_guilt": -0.35,            # each $1 given lowers guilt
    "device_effect_on_happiness": 0.3,
    "keep_effect_on_happiness": 0.12,         # each $1 kept raises happiness
}


def simulate(n: int = 400, seed: int = 2025) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    device = rng.integers(0, 2, n)
    female = rng.integers(0, 2, n)
    econ_major = rng.binomial(1, 0.3, n)
    prosocial = rng.normal(0, 1, n)  # latent trait, unobserved by the analyst

    # Dictator giving: censored to [0, 10], rounded to $0.50 steps.
    give_latent = (2.6 + 0.8 * prosocial + 0.3 * female - 0.5 * econ_major
                   + TRUE["device_effect_on_dictator_give"] * device + rng.normal(0, 1.4, n))
    dictator_give = np.clip(np.round(give_latent * 2) / 2, 0, ENDOWMENT)

    # Ultimatum offer: strategic, anchored near 40% of the pie.
    offer_latent = 3.9 + 0.4 * prosocial - 0.3 * device + rng.normal(0, 0.9, n)
    ultimatum_offer = np.clip(np.round(offer_latent * 2) / 2, 0, ENDOWMENT)
    # Responder acceptance: logistic in the offer share, simulated responders.
    p_accept = 1 / (1 + np.exp(-(ultimatum_offer - 2.8) * 1.6))
    accepted = rng.binomial(1, p_accept)

    guilt = (4.6 + TRUE["give_effect_on_guilt"] * dictator_give
             + TRUE["device_effect_on_guilt"] * device + rng.normal(0, 1.0, n))
    happiness = (3.6 + TRUE["keep_effect_on_happiness"] * (ENDOWMENT - dictator_give)
                 + TRUE["device_effect_on_happiness"] * device + 0.3 * accepted + rng.normal(0, 1.0, n))

    to_scale = lambda x: np.clip(np.round(x), 1, 7).astype(int)  # noqa: E731
    return pd.DataFrame({
        "pid": np.arange(1, n + 1),
        "treatment": np.where(device == 1, "device", "control"),
        "device": device,
        "female": female,
        "econ_major": econ_major,
        "dictator_give": dictator_give,
        "ultimatum_offer": ultimatum_offer,
        "accepted": accepted,
        "guilt": to_scale(guilt),
        "happiness": to_scale(happiness),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    df = simulate(args.n, args.seed)
    df.to_csv(out / "experiment.csv", index=False)
    print(f"wrote {len(df)} simulated participants to {out / 'experiment.csv'}")


if __name__ == "__main__":
    main()
