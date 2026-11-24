"""The benchmark's primary analysis (v0.5-E7a, design §5.5): within-instance paired bootstrap, sign-flip permutation, A/A calibration."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import analysis  # noqa: E402


def _pairs(effect: float, per: int = 30, instances=("9to4", "7to9", "5to8"), seed: int = 0):
    rng = np.random.default_rng(seed)
    return [{"instance": k, "R_L": float(rng.normal(0.4 + effect, 0.3)), "R_R": float(rng.normal(0.4, 0.3))} for k in instances for _ in range(per)]


def test_the_per_instance_table_and_equal_instance_weights():
    pairs = [{"instance": "a", "R_L": 1.0, "R_R": 0.0}] * 3 + [{"instance": "b", "R_L": 0.0, "R_R": 0.0}]  # a: +1 over 3 pairs, b: 0 over 1
    out = analysis.primary(pairs, resamples=200, permutations=200)
    assert out["per_instance"] == [{"instance": "a", "pairs": 3, "both_zero": 0, "delta_R": 1.0},
                                   {"instance": "b", "pairs": 1, "both_zero": 1, "delta_R": 0.0}]  # b's pair: no valid final from either arm
    assert out["delta_R"] == 0.5  # instances weigh equally, not pairs (3 pairs of a would give 0.75)


def test_a_clear_effect_is_found_and_no_effect_is_not():
    found = analysis.primary(_pairs(0.3), resamples=2000, permutations=2000)
    assert found["ci"][0] > 0 and found["p_sign_flip"] < 0.01
    none = analysis.primary(_pairs(0.0, seed=1), resamples=2000, permutations=2000)
    assert none["ci"][0] < 0 < none["ci"][1] and none["p_sign_flip"] > 0.05


def test_the_interval_resamples_within_instances():
    # one instance with zero spread gives an interval of exactly its value: no instance is ever drawn twice or dropped
    pairs = [{"instance": "a", "R_L": 0.7, "R_R": 0.2}] * 5 + [{"instance": "b", "R_L": 0.1, "R_R": 0.1}] * 5
    out = analysis.primary(pairs, resamples=500, permutations=10)
    assert out["ci"] == [pytest.approx(0.25), pytest.approx(0.25)] and out["delta_R"] == pytest.approx(0.25)


def test_aa_calibration_is_near_five_percent():
    # an ITT-like pool: a third of sessions fail (R = 0), the rest spread around 0.6
    rng = np.random.default_rng(3)
    pool = {k: np.where(rng.random(300) < 1 / 3, 0.0, rng.normal(0.6, 0.25, 300)) for k in ("9to4", "7to9", "5to8")}
    rate = analysis.aa_rate(pool, pairs_per_instance=29, studies=300, resamples=400, seed=1)
    assert 0.01 <= rate <= 0.10  # 300 studies: the pre-registered [0.03, 0.07] band is checked at 1,000 in the controls run
