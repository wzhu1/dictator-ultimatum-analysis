import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze import coef_table, fit_models  # noqa: E402
from simulate import ENDOWMENT, simulate  # noqa: E402


def test_simulated_values_in_range():
    df = simulate(500, seed=1)
    assert df.dictator_give.between(0, ENDOWMENT).all()
    assert df.guilt.between(1, 7).all() and df.happiness.between(1, 7).all()
    assert set(df.treatment) == {"control", "device"}


def test_large_sample_recovers_true_effects():
    df = simulate(5000, seed=3)
    table = coef_table(fit_models(df))
    # Binary treatment effects survive the 1-7 rounding/clipping and are recovered.
    assert table[table.term == "device"].covers_truth.all()
    # Continuous slopes are attenuated toward zero by the bounded Likert scale:
    # same sign, significant, but smaller in magnitude than the truth.
    slopes = table[table.term != "device"]
    assert (slopes.p_value < 0.001).all()
    assert (abs(slopes.estimate) < abs(slopes.true_value)).all()
    assert (slopes.estimate * slopes.true_value > 0).all()
