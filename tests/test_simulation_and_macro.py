"""
Tests for the simulation and macro_africa modules.
"""

import json

import numpy as np
import pandas as pd
import pytest
from actuneo.simulation import simulate_aggregate_claims, ClaimsSimulator
from actuneo.loss_reserving import ChainLadder
from actuneo import macro_africa


class TestAggregateClaims:
    def test_compound_poisson_moments(self):
        """Mean = lambda * E[X]; variance = lambda * E[X^2]."""
        result = simulate_aggregate_claims("poisson", [50], "exponential", [1000],
                                           n_simulations=40_000, seed=1)
        assert result.mean == pytest.approx(50_000, rel=0.01)
        assert result.standard_deviation == pytest.approx(np.sqrt(50 * 2 * 1000 ** 2), rel=0.02)
        assert result.counts.mean() == pytest.approx(50, rel=0.01)

    def test_negative_binomial_is_more_variable(self):
        poisson = simulate_aggregate_claims("poisson", [20], "fixed", [100], 20_000, seed=2)
        spread = simulate_aggregate_claims("negative_binomial", [20, 60], "fixed", [100],
                                           20_000, seed=2)
        assert spread.mean == pytest.approx(poisson.mean, rel=0.02)
        assert spread.counts.var() == pytest.approx(60, rel=0.05)
        assert spread.standard_deviation > poisson.standard_deviation

    def test_severity_distributions(self):
        gamma = simulate_aggregate_claims("fixed" if False else "poisson", [1000], "gamma",
                                          [2.0, 500.0], 2_000, seed=3)
        assert gamma.mean == pytest.approx(1000 * 1000, rel=0.01)
        pareto = simulate_aggregate_claims("poisson", [1000], "pareto", [3.0, 2000.0],
                                           2_000, seed=3)
        assert pareto.mean == pytest.approx(1000 * 2000 / 2, rel=0.03)
        lognormal = simulate_aggregate_claims("binomial", [500, 0.1], "lognormal", [7.0, 0.5],
                                              5_000, seed=3)
        assert lognormal.mean == pytest.approx(50 * np.exp(7 + 0.125), rel=0.02)

    def test_retention_and_risk_measures(self):
        gross = simulate_aggregate_claims("poisson", [100], "pareto", [2.5, 3000.0],
                                          20_000, seed=4)
        net = simulate_aggregate_claims("poisson", [100], "pareto", [2.5, 3000.0],
                                        20_000, retention=10_000, seed=4)
        assert net.mean < gross.mean
        assert net.value_at_risk(0.995) < gross.value_at_risk(0.995)
        assert gross.tail_value_at_risk(0.99) >= gross.value_at_risk(0.99)
        assert gross.probability_of_exceeding(gross.value_at_risk(0.9)) == pytest.approx(
            0.1, abs=0.005)
        assert "quantile_99.5%" in gross.summary().index

    def test_reproducible_and_validated(self):
        a = simulate_aggregate_claims("poisson", [10], "exponential", [5], 100, seed=9)
        b = simulate_aggregate_claims("poisson", [10], "exponential", [5], 100, seed=9)
        np.testing.assert_array_equal(a.totals, b.totals)
        with pytest.raises(ValueError):
            simulate_aggregate_claims("geometric", [1], "fixed", [1])
        with pytest.raises(ValueError):
            simulate_aggregate_claims("negative_binomial", [20, 10], "fixed", [1])


class TestClaimsSimulator:
    def test_payments_and_true_ultimate(self):
        simulator = ClaimsSimulator(n_years=6, claims_per_year=200, mean_claim=5000, seed=1)
        payments = simulator.simulate()
        assert set(payments.columns) >= {"claim_id", "accident_year", "payment_year", "amount",
                                         "loss_date", "payment_date"}
        truth = simulator.true_ultimate(payments)
        assert len(truth) == 6
        assert truth.sum() == pytest.approx(payments["amount"].sum())
        # About 200 claims of 5,000 a year
        assert truth.mean() == pytest.approx(1_000_000, rel=0.15)

    def test_triangle_is_what_was_paid_by_the_valuation_date(self):
        simulator = ClaimsSimulator(n_years=6, seed=2)
        payments = simulator.simulate()
        triangle = simulator.triangle(payments)
        assert triangle.shape == (6, 6)
        assert triangle.is_cumulative
        seen = payments[payments["payment_year"] <= 2020]
        assert triangle.latest_diagonal().sum() == pytest.approx(seen["amount"].sum())
        # The same triangle from the dated transactions
        from actuneo.loss_reserving import Triangle
        dated = Triangle.from_transactions(seen, "loss_date", "payment_date", "amount")
        np.testing.assert_allclose(dated.values, triangle.values)

    def test_chain_ladder_recovers_a_stable_pattern(self):
        """With no distortion and many claims, the chain-ladder is close to the truth."""
        simulator = ClaimsSimulator(n_years=10, claims_per_year=3000, claim_cv=0.5, seed=3)
        payments = simulator.simulate()
        estimate = ChainLadder(simulator.triangle(payments)).ultimate.sum()
        truth = simulator.true_ultimate(payments).sum()
        assert estimate == pytest.approx(truth, rel=0.03)

    def test_faster_settlement_misleads_the_chain_ladder(self):
        """Recent years paid sooner look bigger than they are."""
        fast = (0.60, 0.25, 0.10, 0.05)
        simulator = ClaimsSimulator(n_years=10, claims_per_year=3000, claim_cv=0.5, seed=4) \
            .with_settlement_change(2022, fast)
        payments = simulator.simulate()
        estimate = ChainLadder(simulator.triangle(payments)).ultimate.sum()
        truth = simulator.true_ultimate(payments).sum()
        assert estimate > 1.08 * truth

    def test_inflation_and_large_losses(self):
        base = ClaimsSimulator(n_years=5, seed=5).simulate()
        inflated = ClaimsSimulator(n_years=5, inflation=0.10, seed=5).simulate()
        assert inflated["amount"].sum() > base["amount"].sum()
        shocked = ClaimsSimulator(n_years=5, inflation=0.10, seed=5) \
            .with_inflation_shock(2018, 0.50).simulate()
        assert shocked["amount"].sum() > inflated["amount"].sum()
        early = shocked["payment_year"] < 2018
        np.testing.assert_allclose(shocked.loc[early, "amount"], inflated.loc[early, "amount"])

        large = ClaimsSimulator(n_years=5, seed=5).with_large_loss_year(2017, 3.0).simulate()
        ratio = (large.groupby("accident_year")["amount"].sum()
                 / base.groupby("accident_year")["amount"].sum())
        assert ratio.loc[2017] == pytest.approx(3.0)
        assert ratio.loc[2016] == pytest.approx(1.0)


class TestWorldBank:
    """The World Bank reader, with the download replaced by a fixed response."""

    RESPONSE = json.dumps([
        {"page": 1, "pages": 1, "per_page": 1000, "total": 4},
        [
            {"date": "2022", "value": 104.7, "countryiso3code": "ZWE"},
            {"date": "2021", "value": 98.5, "countryiso3code": "ZWE"},
            {"date": "2020", "value": 557.2, "countryiso3code": "ZWE"},
            {"date": "2023", "value": None, "countryiso3code": "ZWE"},
        ],
    ])

    def test_indicator(self):
        requested = []

        def fetch(url):
            requested.append(url)
            return self.RESPONSE

        series = macro_africa.world_bank_indicator("Zimbabwe", "inflation", 2020, 2023, fetch)
        assert "country/ZWE/indicator/FP.CPI.TOTL.ZG" in requested[0]
        assert "date=2020:2023" in requested[0]
        assert series.index.tolist() == [2020, 2021, 2022]   # the missing year is left out
        assert series.loc[2020] == 557.2
        assert series.index.name == "year"

    def test_codes_pass_through(self):
        requested = []
        macro_africa.world_bank_indicator(
            "ken", "SP.POP.TOTL", fetch=lambda url: requested.append(url) or self.RESPONSE)
        assert "country/KEN/indicator/SP.POP.TOTL" in requested[0]

    def test_several_indicators(self):
        frame = macro_africa.macro_data("Zimbabwe", ["inflation", "gdp_growth"],
                                        fetch=lambda url: self.RESPONSE)
        assert list(frame.columns) == ["inflation", "gdp_growth"]
        assert frame.loc[2021, "inflation"] == 98.5

    def test_no_data_and_no_connection(self):
        empty = json.dumps([{"message": [{"id": "120", "value": "Invalid value"}]}])
        with pytest.raises(ValueError, match="no data"):
            macro_africa.world_bank_indicator("Zimbabwe", "inflation", fetch=lambda url: empty)

        def offline(url):
            raise OSError("network unreachable")

        with pytest.raises(ConnectionError, match="internet connection"):
            macro_africa.world_bank_indicator("Zimbabwe", "inflation", fetch=offline)
        frame = macro_africa.macro_data("Zimbabwe", ["inflation"], fetch=lambda url: empty)
        assert frame.empty

    def test_reference_lists(self):
        assert macro_africa.COUNTRIES["Zimbabwe"] == "ZWE"
        assert macro_africa.INDICATORS["inflation"] == "FP.CPI.TOTL.ZG"
