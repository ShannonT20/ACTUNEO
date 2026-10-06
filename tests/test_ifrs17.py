"""
Tests for the ifrs17 module: premium allocation approach, liability for
incurred claims, reinsurance held and the financial statements.

The liability for remaining coverage is checked against the IASB's published
premium allocation approach example (Transition Resource Group, May 2018):
a premium of 1,200 for four quarters of cover with acquisition cash flows
of 180.
"""

import numpy as np
import pandas as pd
import pytest
from actuneo.finance import YieldCurve
from actuneo.loss_reserving import (
    Triangle, ChainLadder, MackChainLadder, BootChainLadder, load_raa, load_genins,
)
from actuneo.ifrs17 import (
    PAAGroup, PAAReinsuranceHeld, paa_eligibility, LiabilityForIncurredClaims,
    lic_analysis_of_change, risk_adjustment_confidence_level,
    risk_adjustment_cost_of_capital, implied_confidence_level, IFRS17Statements,
    earned_fraction, unearned_premium, earned_premium_by_period,
)

QUARTERS = ["30.09.X1", "31.12.X1", "31.03.X2", "30.06.X2"]


@pytest.fixture
def group():
    """A group with claims, risk adjustment, discount unwind and an onerous period."""
    return PAAGroup(
        premiums_received=[1200, 0, 0, 0],
        acquisition_cash_flows=[180, 0, 0, 0],
        fulfilment_cash_flows_remaining=[800, 560, 240, 0],
        incurred_claims=[200, 210, 190, 220],
        incurred_risk_adjustment=[10, 10, 10, 10],
        claims_paid=[100, 180, 200, 150],
        closing_lic_pv=[100, 125, 120, 185],
        closing_lic_ra=[10, 14, 12, 15],
        finance_expenses=[0, 2, 3, 3],
        periods=QUARTERS, name="Motor 20X1", portfolio="Motor",
    )


@pytest.fixture
def reinsurance():
    return PAAReinsuranceHeld(
        premiums_paid=[240, 0, 0, 0],
        recoveries_incurred=[40, 42, 38, 44],
        recoveries_received=[20, 36, 40, 30],
        loss_recovery_component=[7, 2, 0, 0],
        periods=QUARTERS, name="Motor quota share",
    )


class TestLiabilityForRemainingCoverage:
    """The IASB example and variations on it."""

    def test_premium_received_at_initial_recognition(self):
        g = PAAGroup([1200, 0, 0, 0], acquisition_cash_flows=[180, 0, 0, 0], periods=QUARTERS)
        np.testing.assert_allclose(g.insurance_revenue, 300)
        np.testing.assert_allclose(g.amortisation, 45)
        np.testing.assert_allclose(g.lrc, [765, 510, 255, 0])

        roll = g.lrc_rollforward()
        assert roll.loc["30.09.X1", "opening"] == 0
        assert roll.loc["31.12.X1", "opening"] == 765
        np.testing.assert_allclose(
            roll.drop(columns="closing").sum(axis=1), roll["closing"]
        )

    def test_premium_received_at_end_of_coverage(self):
        """Nothing is received until the end, so the group is an asset until then."""
        g = PAAGroup([0, 0, 0, 1200], acquisition_cash_flows=[180, 0, 0, 0])
        np.testing.assert_allclose(g.lrc, [-435, -690, -945, 0])

    def test_premium_received_in_instalments(self):
        g = PAAGroup([300] * 4, expected_premium=1200, acquisition_cash_flows=[180, 0, 0, 0])
        np.testing.assert_allclose(g.lrc, [-135, -90, -45, 0])

    def test_all_three_cases_report_the_same_result(self):
        """Timing of the premium changes the balance sheet, not the profit."""
        results = [
            PAAGroup(p, expected_premium=1200, acquisition_cash_flows=[180, 0, 0, 0])
            .profit_or_loss()["insurance_service_result"].to_numpy()
            for p in ([1200, 0, 0, 0], [0, 0, 0, 1200], [300] * 4)
        ]
        for result in results:
            np.testing.assert_allclose(result, 300 - 45)

    def test_acquisition_cash_flows_expensed(self):
        g = PAAGroup([1200, 0, 0, 0], acquisition_cash_flows=[180, 0, 0, 0],
                     expense_acquisition_cash_flows=True)
        np.testing.assert_allclose(g.lrc, [900, 600, 300, 0])
        np.testing.assert_allclose(
            g.profit_or_loss()["insurance_service_result"], [120, 300, 300, 300]
        )
        # Total profit is the same as when the cash flows are deferred
        assert g.profit_or_loss()["insurance_service_result"].sum() == 1200 - 180

    def test_acquisition_cash_flows_paid_later(self):
        """A cost paid half way through is spread over the cover still to be given."""
        g = PAAGroup([1200, 0, 0, 0], acquisition_cash_flows=[100, 0, 80, 0])
        # 100 over four quarters, then the unamortised 50 plus 80 over the last two
        np.testing.assert_allclose(g.amortisation, [25, 25, 65, 65])
        np.testing.assert_allclose(g.deferred_acquisition_cash_flows, [75, 50, 65, 0])
        assert g.amortisation.sum() == 180
        assert g.lrc[-1] == pytest.approx(0)

    def test_revenue_pattern(self):
        """Revenue following the expected pattern of claims instead of time."""
        g = PAAGroup([1200, 0, 0, 0], revenue_pattern=[0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(g.insurance_revenue, [120, 240, 360, 480])
        np.testing.assert_allclose(g.lrc, [1080, 840, 480, 0])

    def test_cover_continuing_after_the_last_period(self):
        """Half of a one-year group has been earned after two quarters."""
        g = PAAGroup([1200, 0], revenue_pattern=[0.25, 0.25], acquisition_cash_flows=[180, 0])
        np.testing.assert_allclose(g.insurance_revenue, [300, 300])
        np.testing.assert_allclose(g.lrc, [765, 510])

    def test_explicit_revenue(self):
        g = PAAGroup([1000, 200], insurance_revenue=[500, 700])
        np.testing.assert_allclose(g.lrc, [500, 0])
        with pytest.raises(ValueError, match="either insurance_revenue or revenue_pattern"):
            PAAGroup([1000, 200], insurance_revenue=[500, 700], revenue_pattern=[0.5, 0.5])

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="one value for each of the 4 periods"):
            PAAGroup([1200, 0, 0, 0], acquisition_cash_flows=[180, 0])
        with pytest.raises(ValueError, match="one label"):
            PAAGroup([1200, 0], periods=["a"])


class TestOnerousContracts:
    """Loss component of the liability for remaining coverage."""

    def test_loss_component(self, group):
        # Carrying amounts 765, 510, 255, 0 against fulfilment cash flows 800, 560, 240, 0
        np.testing.assert_allclose(group.loss_component, [35, 50, 0, 0])
        np.testing.assert_allclose(group.loss_on_onerous_contracts, [35, 15, -50, 0])
        np.testing.assert_allclose(group.lrc, [800, 560, 255, 0])

    def test_loss_and_reversal_in_profit_or_loss(self, group):
        losses = group.profit_or_loss()["losses_on_onerous_contracts"]
        np.testing.assert_allclose(losses, [-35, -15, 50, 0])
        # The loss reverses in full by the end of the cover
        assert losses.sum() == pytest.approx(0)

    def test_not_onerous(self):
        g = PAAGroup([1200, 0, 0, 0], fulfilment_cash_flows_remaining=[700, 400, 200, 0])
        np.testing.assert_allclose(g.loss_component, 0)


class TestGroupArticulation:
    """Balances, profit and cash tie together."""

    def test_change_in_liability_is_cash_less_profit(self, group):
        profit = group.profit_or_loss()
        result = profit["insurance_service_result"] + profit["insurance_finance_expenses"]
        cash = group.cash_flows()["net_cash_flow"]
        change = np.diff(np.concatenate(([0.0], group.carrying_amount)))
        np.testing.assert_allclose(change, cash - result)

    def test_expensed_acquisition_cash_flows_also_tie(self):
        g = PAAGroup([1200, 0, 0, 0], acquisition_cash_flows=[180, 0, 0, 0],
                     expense_acquisition_cash_flows=True, incurred_claims=[250] * 4,
                     claims_paid=[100, 200, 300, 400])
        profit = g.profit_or_loss()["insurance_service_result"]
        change = np.diff(np.concatenate(([0.0], g.carrying_amount)))
        np.testing.assert_allclose(change, g.cash_flows()["net_cash_flow"] - profit)

    def test_lic_rollforward(self, group):
        roll = group.lic_rollforward()
        np.testing.assert_allclose(roll.drop(columns="closing").sum(axis=1), roll["closing"])
        np.testing.assert_allclose(roll["closing"], [110, 139, 132, 200])
        # Estimates of earlier claims were reduced in the second quarter
        assert roll.loc["31.12.X1", "adjustments_to_lic"] == pytest.approx(-13)

    def test_default_lic_has_no_change_in_estimates(self):
        g = PAAGroup([1200, 0], incurred_claims=[300, 320], incurred_risk_adjustment=[15, 16],
                     claims_paid=[100, 250], finance_expenses=[0, 4])
        np.testing.assert_allclose(g.lic_present_value, [200, 274])
        np.testing.assert_allclose(g.lic_risk_adjustment, [15, 31])
        np.testing.assert_allclose(g.adjustments_to_lic_pv, 0)

    def test_profit_or_loss(self, group):
        profit = group.profit_or_loss()
        np.testing.assert_allclose(profit["insurance_revenue"], 300)
        np.testing.assert_allclose(profit["insurance_service_expenses"], [-290, -267, -185, -260])
        np.testing.assert_allclose(profit["insurance_service_result"], [10, 33, 115, 40])
        np.testing.assert_allclose(profit["insurance_finance_expenses"], [0, -2, -3, -3])

    def test_reconciliation(self, group):
        table = group.reconciliation("31.12.X1")
        assert list(table.columns) == ["lrc_excluding_loss_component", "loss_component",
                                       "lic_present_value", "lic_risk_adjustment", "total"]
        np.testing.assert_allclose(table.loc["Opening liabilities"], [765, 35, 100, 10, 910])
        np.testing.assert_allclose(table.loc["Closing liabilities"], [510, 50, 125, 14, 699])
        np.testing.assert_allclose(table.loc["Insurance service result"], [-255, 15, 203, 4, -33])
        # Opening + changes in profit or loss + cash flows = closing, in every column
        closing = (table.loc["Opening liabilities"] + table.loc["Total changes in profit or loss"]
                   + table.loc["Total cash flows"])
        np.testing.assert_allclose(closing, table.loc["Closing liabilities"])

    def test_reconciliation_ties_to_profit_or_loss(self, group):
        for period in QUARTERS:
            table = group.reconciliation(period)
            profit = group.profit_or_loss().loc[period]
            assert table.loc["Insurance service result", "total"] == pytest.approx(
                -profit["insurance_service_result"]
            )
        # The default is the last period, which starts from the third quarter's balances
        np.testing.assert_allclose(
            group.reconciliation().loc["Opening liabilities"], [255, 0, 120, 12, 387]
        )


class TestEligibility:
    """PAA eligibility."""

    def test_one_year_or_less(self):
        result = paa_eligibility(12)
        assert result["eligible"] and "one year or less" in result["basis"]

    def test_longer_contracts_need_projections(self):
        assert not paa_eligibility(24)["eligible"]
        close = paa_eligibility(24, lrc_paa=[100, 50, 20], lrc_general_model=[102, 51, 20])
        assert close["eligible"]
        assert close["max_relative_difference"] == pytest.approx(2 / 102)
        far = paa_eligibility(36, [100, 50, 20], [120, 70, 20])
        assert not far["eligible"]
        assert far["max_relative_difference"] == pytest.approx(20 / 70)
        assert paa_eligibility(36, [100, 50], [120, 70], threshold=0.5)["eligible"]


class TestLiabilityForIncurredClaims:
    """Discounting and risk adjustment of incurred claims."""

    def test_discounting_by_hand(self):
        lic = LiabilityForIncurredClaims([100, 50], discount_rate=0.10, risk_adjustment=12)
        assert lic.undiscounted == 150
        assert lic.present_value == pytest.approx(100 * 1.1 ** -0.5 + 50 * 1.1 ** -1.5)
        assert lic.total == pytest.approx(lic.present_value + 12)
        assert lic.discounting == pytest.approx(150 - lic.present_value)
        end = LiabilityForIncurredClaims([100, 50], 0.10, timing=1)
        assert end.present_value == pytest.approx(100 / 1.1 + 50 / 1.21)

    def test_no_discounting(self):
        lic = LiabilityForIncurredClaims([100, 50])
        assert lic.present_value == 150 and lic.discounting == 0

    def test_quarterly_periods_and_curve(self):
        quarterly = LiabilityForIncurredClaims([100, 50], 0.10, periods_per_year=4)
        assert quarterly.present_value == pytest.approx(100 * 1.1 ** -0.125 + 50 * 1.1 ** -0.375)
        flat = YieldCurve([1, 5], [0.10, 0.10])
        assert LiabilityForIncurredClaims([100, 50], flat).present_value == pytest.approx(
            LiabilityForIncurredClaims([100, 50], 0.10).present_value
        )

    def test_from_reserving_model(self):
        mack = MackChainLadder(load_raa(), est_sigma="mack")
        lic = LiabilityForIncurredClaims.from_reserving(mack, discount_rate=0.08,
                                                        confidence_level=0.75)
        assert lic.undiscounted == pytest.approx(mack.total_ibnr)
        assert lic.present_value == pytest.approx(mack.discounted_reserve(0.08))
        undiscounted_ra = risk_adjustment_confidence_level(mack, 0.75)
        # The risk adjustment is discounted in proportion to the cash flows
        assert lic.risk_adjustment == pytest.approx(
            undiscounted_ra * lic.present_value / lic.undiscounted
        )
        assert "risk_adjustment" in lic.summary().index

    def test_from_reserving_without_uncertainty(self):
        cl = ChainLadder(load_raa())
        lic = LiabilityForIncurredClaims.from_reserving(cl, risk_adjustment=5000)
        assert lic.present_value == pytest.approx(cl.total_ibnr)
        assert lic.risk_adjustment == pytest.approx(5000)
        with pytest.raises(TypeError):
            LiabilityForIncurredClaims.from_reserving(cl, confidence_level=0.75)
        with pytest.raises(ValueError, match="either"):
            LiabilityForIncurredClaims.from_reserving(cl, confidence_level=0.75,
                                                      risk_adjustment=1.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            LiabilityForIncurredClaims([100], risk_adjustment=-1)
        with pytest.raises(ValueError):
            LiabilityForIncurredClaims([100], timing=1.5)


class TestRiskAdjustment:
    """Confidence level and cost of capital techniques."""

    def test_confidence_level_from_mack(self):
        mack = MackChainLadder(load_genins(), est_sigma="mack")
        ra75 = risk_adjustment_confidence_level(mack, 0.75)
        ra90 = risk_adjustment_confidence_level(mack, 0.90)
        expected = mack.reserve_quantile(0.75).loc["Total"].iloc[0] - mack.total_ibnr
        assert ra75 == pytest.approx(expected)
        assert 0 < ra75 < ra90
        normal = risk_adjustment_confidence_level(mack, 0.75, distribution="normal")
        assert normal == pytest.approx(0.6744898 * mack.total_mack_se, rel=1e-6)

    def test_never_negative(self):
        """Below the mean of a skewed distribution the margin is floored at zero."""
        mack = MackChainLadder(load_raa(), est_sigma="mack")
        assert risk_adjustment_confidence_level(mack, 0.30) == 0.0

    def test_confidence_level_from_bootstrap(self):
        boot = BootChainLadder(load_genins(), 3000, seed=5)
        ra = risk_adjustment_confidence_level(boot, 0.75)
        assert ra == pytest.approx(boot.total_ibnr_simulations.quantile(0.75) - boot.mean_ibnr)
        mack = MackChainLadder(load_genins(), est_sigma="mack")
        # Same order of magnitude as the Mack lognormal margin; the bootstrap is wider
        assert 0.8 < ra / risk_adjustment_confidence_level(mack, 0.75) < 1.6

    def test_implied_confidence_level_round_trip(self):
        mack = MackChainLadder(load_genins(), est_sigma="mack")
        for distribution in ("lognormal", "normal"):
            ra = risk_adjustment_confidence_level(mack, 0.85, distribution)
            assert implied_confidence_level(mack, ra, distribution) == pytest.approx(0.85)
        boot = BootChainLadder(load_genins(), 2000, seed=5)
        ra = risk_adjustment_confidence_level(boot, 0.80)
        assert implied_confidence_level(boot, ra) == pytest.approx(0.80, abs=0.002)

    def test_cost_of_capital_by_hand(self):
        ra = risk_adjustment_cost_of_capital([1000, 600, 200], cost_of_capital=0.06,
                                             discount_rate=0.10)
        assert ra == pytest.approx(60 / 1.1 + 36 / 1.21 + 12 / 1.331)
        assert risk_adjustment_cost_of_capital([1000, 600, 200], 0.06) == pytest.approx(108)
        with pytest.raises(TypeError):
            risk_adjustment_confidence_level(ChainLadder(load_raa()), 0.75)
        with pytest.raises(ValueError):
            risk_adjustment_confidence_level(MackChainLadder(load_raa()), 1.2)


class TestLicAnalysisOfChange:
    """Splitting the movement of the liability for incurred claims."""

    def test_identity(self):
        change = lic_analysis_of_change(
            opening_cash_flows=[400, 250, 100],
            closing_cash_flows_prior=[270, 120],
            closing_cash_flows_current=[300, 150, 50],
            paid_prior=380, paid_current=200,
            opening_rate=0.08, closing_rate=0.10,
            opening_risk_adjustment=40, closing_risk_adjustment_prior=22,
            closing_risk_adjustment_current=30,
        )
        closing = (change["opening_lic_pv"] + change["incurred_claims"]
                   + change["adjustments_to_lic_pv"] + change["finance_expenses"]
                   - change["claims_paid"])
        assert closing == pytest.approx(change["closing_lic_pv"])
        assert change["closing_lic_ra"] == 52
        assert change["adjustments_to_lic_ra"] == -18
        assert change["finance_expenses"] == pytest.approx(
            change["interest_accreted"] + change["effect_of_rate_changes"]
        )

    def test_no_surprises_means_only_interest(self):
        """Claims paid and re-estimated exactly as expected, at an unchanged rate."""
        change = lic_analysis_of_change([400, 250, 100], [250, 100], [], 400, 0,
                                        opening_rate=0.08)
        assert change["adjustments_to_lic_pv"] == pytest.approx(0)
        assert change["effect_of_rate_changes"] == pytest.approx(0)
        assert change["incurred_claims"] == 0
        # Interest: every payment is one year nearer, the first is paid in the middle of the year
        opening = 400 * 1.08 ** -0.5 + 250 * 1.08 ** -1.5 + 100 * 1.08 ** -2.5
        end = 400 + 250 * 1.08 ** -0.5 + 100 * 1.08 ** -1.5
        assert change["interest_accreted"] == pytest.approx(end - opening)
        assert change["interest_accreted"] > 0

    def test_undiscounted(self):
        change = lic_analysis_of_change([400, 250], [300], [500, 100], 420, 150)
        assert change["finance_expenses"] == 0
        assert change["incurred_claims"] == 750
        # Expected to pay 400 and hold 250; paid 420 and holds 300
        assert change["adjustments_to_lic_pv"] == pytest.approx(70)

    def test_rate_rise_reduces_the_liability(self):
        change = lic_analysis_of_change([400, 250, 100], [250, 100], [], 400, 0, 0.08, 0.12)
        assert change["effect_of_rate_changes"] < 0

    def test_from_two_valuations_of_a_triangle(self):
        """Opening and closing reserving models feed the PAA group directly."""
        raa = load_raa()
        closing_model = ChainLadder(raa)
        # The triangle as it stood one year earlier
        earlier = raa.values[:9, :9].copy()
        for i in range(9):
            earlier[i, 9 - i:] = np.nan
        opening_model = ChainLadder(Triangle(earlier, raa.origin[:9], raa.development[:9]))

        flows = closing_model.cash_flows_by_origin()
        np.testing.assert_allclose(flows.sum(axis=0), closing_model.cash_flows())
        incremental = raa.to_incremental().values
        latest = [incremental[i, 9 - i] for i in range(10)]

        change = lic_analysis_of_change(
            opening_cash_flows=opening_model.cash_flows(),
            closing_cash_flows_prior=flows.iloc[:9].sum(axis=0),
            closing_cash_flows_current=flows.iloc[9],
            paid_prior=sum(latest[:9]), paid_current=latest[9],
            opening_rate=0.07, closing_rate=0.07,
        )
        assert change["claims_paid"] == pytest.approx(sum(latest))
        assert change["closing_lic_pv"] == pytest.approx(closing_model.discounted_reserve(0.07))
        assert change["opening_lic_pv"] == pytest.approx(opening_model.discounted_reserve(0.07))

        group = PAAGroup(
            premiums_received=[0.0],
            incurred_claims=[change["incurred_claims"]],
            claims_paid=[change["claims_paid"]],
            finance_expenses=[change["finance_expenses"]],
            closing_lic_pv=[change["closing_lic_pv"]],
        )
        # Started from nil, so the opening liability shows up as an adjustment
        assert group.lic_present_value[0] == pytest.approx(change["closing_lic_pv"])


class TestReinsuranceHeld:
    """Reinsurance contracts held under the PAA."""

    def test_balances(self, reinsurance):
        np.testing.assert_allclose(reinsurance.allocation_of_premiums, 60)
        np.testing.assert_allclose(reinsurance.arc_excluding_loss_recovery, [180, 120, 60, 0])
        np.testing.assert_allclose(reinsurance.aic_present_value, [20, 26, 24, 38])
        np.testing.assert_allclose(reinsurance.carrying_amount, [207, 148, 84, 38])

    def test_profit_or_loss(self, reinsurance):
        profit = reinsurance.profit_or_loss()
        np.testing.assert_allclose(profit["allocation_of_reinsurance_premiums"], -60)
        np.testing.assert_allclose(profit["recoveries_of_losses_on_onerous_contracts"],
                                   [7, -5, -2, 0])
        np.testing.assert_allclose(profit["net_expenses_from_reinsurance_contracts"],
                                   [-13, -23, -24, -16])

    def test_articulation(self, reinsurance):
        """Change in the asset = result in profit or loss less net cash received."""
        profit = reinsurance.profit_or_loss()
        result = (profit["net_expenses_from_reinsurance_contracts"]
                  + profit["reinsurance_finance_income"])
        cash = reinsurance.cash_flows()["net_cash_flow"]
        change = np.diff(np.concatenate(([0.0], reinsurance.carrying_amount)))
        np.testing.assert_allclose(change, result - cash)

    def test_loss_recovery_for_quota_share(self, group):
        """A 20% quota share recovers 20% of the loss on the onerous underlying group."""
        ceded = PAAReinsuranceHeld([240, 0, 0, 0],
                                   loss_recovery_component=0.2 * group.loss_component)
        np.testing.assert_allclose(ceded.loss_recovery_income,
                                   0.2 * group.loss_on_onerous_contracts)
        with pytest.raises(ValueError, match="negative"):
            PAAReinsuranceHeld([240, 0], loss_recovery_component=[-1, 0])


class TestFinancialStatements:
    """Statement of profit or loss, financial position and cash flows."""

    @pytest.fixture
    def statements(self, group, reinsurance):
        return IFRS17Statements(
            [group], [reinsurance], opening_equity=500,
            investment_return=[10, 12, 12, 13], other_operating_expenses=20, tax_rate=0.25,
        )

    def test_profit_or_loss(self, statements):
        profit = statements.profit_or_loss()
        assert list(profit.columns) == QUARTERS
        np.testing.assert_allclose(profit.loc["Insurance revenue"], 300)
        np.testing.assert_allclose(profit.loc["Insurance service result"], [-3, 10, 91, 24])
        np.testing.assert_allclose(profit.loc["Net financial result"], [10, 10, 9, 10])
        np.testing.assert_allclose(profit.loc["Profit before tax"], [-13, 0, 80, 14])
        # No tax on a loss
        np.testing.assert_allclose(profit.loc["Income tax expense"], [0, 0, -20, -3.5])
        np.testing.assert_allclose(profit.loc["Profit for the period"], [-13, 0, 60, 10.5])

    def test_financial_position_balances(self, statements):
        position = statements.financial_position()
        np.testing.assert_allclose(position.loc["Total assets"],
                                   position.loc["Total liabilities and equity"])
        assert statements.balance_check() == pytest.approx(0, abs=1e-9)
        np.testing.assert_allclose(position.loc["Insurance contract liabilities"],
                                   [910, 699, 387, 200])
        np.testing.assert_allclose(position.loc["Reinsurance contract assets"],
                                   [207, 148, 84, 38])
        np.testing.assert_allclose(position.loc["Total equity"], [487, 487, 547, 557.5])

    def test_cash_flows(self, statements):
        cash = statements.cash_flows()
        np.testing.assert_allclose(cash.loc["Net increase in cash and investments"],
                                   [690, -152, -188, -130.5])
        assert cash.loc["Cash and investments at start of period", "30.09.X1"] == 500
        np.testing.assert_allclose(
            cash.loc["Cash and investments at end of period"],
            statements.financial_position().loc["Cash and investments"],
        )

    def test_equity_is_opening_plus_profits(self, statements):
        profit = statements.profit_or_loss().loc["Profit for the period"].cumsum()
        equity = statements.financial_position().loc["Total equity"]
        np.testing.assert_allclose(equity, 500 + profit)

    def test_portfolios_in_asset_and_liability_positions(self):
        """Portfolios are not offset: one is an asset, the other a liability."""
        asset = PAAGroup([0, 0, 0, 1200], expected_premium=1200, name="A", portfolio="Fire")
        liability = PAAGroup([1200, 0, 0, 0], name="B", portfolio="Motor")
        position = IFRS17Statements([asset, liability]).financial_position()
        np.testing.assert_allclose(position.loc["Insurance contract assets"], [300, 600, 900, 0])
        np.testing.assert_allclose(position.loc["Insurance contract liabilities"],
                                   [900, 600, 300, 0])
        assert IFRS17Statements([asset, liability]).balance_check() == pytest.approx(0)

    def test_groups_of_one_portfolio_are_netted(self):
        asset = PAAGroup([0, 0, 0, 1200], expected_premium=1200, name="A", portfolio="Motor")
        liability = PAAGroup([1200, 0, 0, 0], name="B", portfolio="Motor")
        position = IFRS17Statements([asset, liability]).financial_position()
        # Group balances -300, -600, -900, 0 and 900, 600, 300, 0 net to 600, 0, -600, 0
        np.testing.assert_allclose(position.loc["Insurance contract liabilities"],
                                   [600, 0, 0, 0])
        np.testing.assert_allclose(position.loc["Insurance contract assets"], [0, 0, 600, 0])

    def test_key_ratios(self, statements):
        ratios = statements.key_ratios()
        np.testing.assert_allclose(ratios.loc["Acquisition expense ratio"], 0.15)
        np.testing.assert_allclose(ratios.loc["Other expense ratio"], 20 / 300)
        np.testing.assert_allclose(ratios.loc["Claims ratio"],
                                   [245 / 300, 222 / 300, 140 / 300, 215 / 300])
        profit = statements.profit_or_loss()
        # Combined ratio above 100% goes with an underwriting loss after other expenses
        underwriting = (profit.loc["Insurance service result"]
                        + profit.loc["Other operating expenses"])
        np.testing.assert_allclose(1 - ratios.loc["Combined ratio"], underwriting / 300)

    def test_invalid(self, group):
        with pytest.raises(ValueError, match="same reporting periods"):
            IFRS17Statements([group, PAAGroup([100, 0])])
        with pytest.raises(ValueError, match="At least one group"):
            IFRS17Statements([])
        with pytest.raises(ValueError, match="tax_rate"):
            IFRS17Statements([group], tax_rate=1.5)


class TestPremiumEarning:
    """Earning premiums by days of cover."""

    @pytest.fixture
    def policies(self):
        return pd.DataFrame({
            "premium": [365.0, 730.0, 120.0],
            "start_date": ["2025-01-01", "2025-07-01", "2025-11-01"],
            "end_date": ["2025-12-31", "2026-06-30", "2025-11-30"],
        })

    def test_earned_fraction(self):
        fraction = earned_fraction(["2025-01-01"] * 4, ["2025-12-31"] * 4,
                                   "2025-01-01")
        np.testing.assert_allclose(fraction, 1 / 365)
        np.testing.assert_allclose(
            earned_fraction(["2025-01-01", "2025-07-01"], ["2025-12-31", "2026-06-30"],
                            "2025-06-30"),
            [181 / 365, 0],
        )
        # Before cover starts and after it ends
        assert earned_fraction(["2025-03-01"], ["2025-03-31"], "2025-02-01")[0] == 0
        assert earned_fraction(["2025-03-01"], ["2025-03-31"], "2026-01-01")[0] == 1

    def test_unearned_premium(self, policies):
        unearned = unearned_premium(policies["premium"], policies["start_date"],
                                    policies["end_date"], "2025-12-31")
        np.testing.assert_allclose(unearned, [0, 730 * 181 / 365, 0])

    def test_earned_by_period(self, policies):
        earned = earned_premium_by_period(policies, ["2025-06-30", "2025-12-31", "2026-06-30"])
        np.testing.assert_allclose(earned, [181, 184 + 730 * 184 / 365 + 120, 730 * 181 / 365])
        assert earned.sum() == pytest.approx(policies["premium"].sum())

    def test_earned_plus_unearned_is_written(self, policies):
        earned = earned_premium_by_period(policies, ["2025-06-30", "2025-12-31"])
        unearned = unearned_premium(policies["premium"], policies["start_date"],
                                    policies["end_date"], "2025-12-31")
        assert earned.sum() + unearned.sum() == pytest.approx(policies["premium"].sum())

    def test_feeds_the_paa_group(self, policies):
        """With premiums received up front, the liability is the unearned premium."""
        dates = ["2025-06-30", "2025-12-31", "2026-06-30"]
        earned = earned_premium_by_period(policies, dates)
        received = [365.0, 850.0, 0.0]
        group = PAAGroup(received, insurance_revenue=earned, periods=dates)
        unearned = unearned_premium(policies["premium"], policies["start_date"],
                                    policies["end_date"], "2025-12-31").sum()
        assert group.lrc[1] == pytest.approx(unearned)
        assert group.lrc[2] == pytest.approx(0)

    def test_invalid(self, policies):
        with pytest.raises(ValueError, match="increasing"):
            earned_premium_by_period(policies, ["2025-12-31", "2025-06-30"])
        with pytest.raises(ValueError, match="ends before"):
            earned_fraction(["2025-05-01"], ["2025-04-01"], "2025-06-01")


class TestGroupsAlreadyInForce:
    """Opening balances for a group that is part way through its cover."""

    def test_second_half_of_the_iasb_example(self):
        """Starting from the half-year balances gives the same last two quarters."""
        g = PAAGroup([0, 0], expected_premium=600, acquisition_cash_flows=0,
                     opening_lrc=510, opening_deferred_acquisition_cash_flows=90)
        np.testing.assert_allclose(g.insurance_revenue, 300)
        np.testing.assert_allclose(g.amortisation, 45)
        np.testing.assert_allclose(g.lrc, [255, 0])
        assert g.lrc_rollforward().loc[1, "opening"] == 510
        assert g.opening_carrying_amount == 510

    def test_opening_claims_liability_and_loss_component(self):
        g = PAAGroup([0, 0], expected_premium=600, opening_lrc=600, opening_loss_component=40,
                     fulfilment_cash_flows_remaining=[320, 0],
                     opening_lic_pv=500, opening_lic_ra=30,
                     incurred_claims=[200, 210], claims_paid=[260, 240],
                     closing_lic_pv=[430, 410], closing_lic_ra=[26, 22])
        # Loss component 40 -> 20 -> 0: reversals of 20 each period
        np.testing.assert_allclose(g.loss_on_onerous_contracts, [-20, -20])
        # Claims: 500 + 200 - 260 = 440 expected, 430 held
        np.testing.assert_allclose(g.adjustments_to_lic_pv, [-10, 10])
        np.testing.assert_allclose(g.adjustments_to_lic_ra, [-4, -4])
        table = g.reconciliation(1)
        np.testing.assert_allclose(table.loc["Opening liabilities"], [600, 40, 500, 30, 1170])
        closing = (table.loc["Opening liabilities"] + table.loc["Total changes in profit or loss"]
                   + table.loc["Total cash flows"])
        np.testing.assert_allclose(closing, table.loc["Closing liabilities"])

    def test_statements_balance_from_opening_balances(self):
        g = PAAGroup([0, 0], expected_premium=600, opening_lrc=600,
                     opening_lic_pv=500, incurred_claims=[200, 210], claims_paid=[260, 240])
        r = PAAReinsuranceHeld([0, 0], expected_premium=120, opening_arc=120,
                               opening_aic_pv=100, recoveries_incurred=[40, 42],
                               recoveries_received=[52, 48])
        statements = IFRS17Statements([g], [r], opening_equity=400)
        # Cash backs the equity and the net insurance liabilities
        assert statements.opening_cash == 400 + 1100 - 220
        assert statements.balance_check() == pytest.approx(0, abs=1e-9)
        equity = statements.changes_in_equity()
        assert equity.loc["Equity at start of period", 1] == 400


class TestOtherExpensesAndOci:
    """Other attributable expenses and the OCI option for finance expenses."""

    @pytest.fixture
    def oci_group(self):
        return PAAGroup(
            [1200, 0, 0, 0], acquisition_cash_flows=[180, 0, 0, 0],
            incurred_claims=[200, 210, 190, 220], claims_paid=[100, 180, 200, 150],
            finance_expenses=[0, 6, 5, 4], finance_expenses_in_oci=[0, 4, 2, 1],
            other_insurance_service_expenses=[12, 12, 12, 12], periods=QUARTERS,
        )

    def test_other_expenses(self, oci_group):
        profit = oci_group.profit_or_loss()
        np.testing.assert_allclose(profit["incurred_claims_and_other_expenses"],
                                   [-212, -222, -202, -232])
        np.testing.assert_allclose(oci_group.cash_flows()["claims_paid"],
                                   [-112, -192, -212, -162])
        # Paid when incurred, so they pass through the claims liability without a balance
        table = oci_group.reconciliation("31.12.X1")
        assert table.loc["Incurred claims and other insurance service expenses",
                         "lic_present_value"] == 222
        assert table.loc["Claims and other insurance service expenses paid",
                         "lic_present_value"] == -192

    def test_finance_expenses_split(self, oci_group):
        profit = oci_group.profit_or_loss()
        np.testing.assert_allclose(profit["insurance_finance_expenses"], [0, -2, -3, -3])
        np.testing.assert_allclose(profit["insurance_finance_expenses_oci"], [0, -4, -2, -1])

    def test_comprehensive_income_and_reserve(self, oci_group):
        statements = IFRS17Statements([oci_group], opening_equity=300)
        performance = statements.profit_or_loss()
        np.testing.assert_allclose(performance.loc["Other comprehensive income"],
                                   [0, -4, -2, -1])
        np.testing.assert_allclose(
            performance.loc["Total comprehensive income"],
            performance.loc["Profit for the period"]
            + performance.loc["Other comprehensive income"],
        )
        position = statements.financial_position()
        np.testing.assert_allclose(position.loc["Insurance finance reserve"], [0, -4, -6, -7])
        assert statements.balance_check() == pytest.approx(0, abs=1e-9)

        equity = statements.changes_in_equity()
        np.testing.assert_allclose(equity.loc["Equity at end of period"],
                                   position.loc["Total equity"])
        np.testing.assert_allclose(equity.loc["Equity at start of period"].iloc[1:],
                                   equity.loc["Equity at end of period"].iloc[:-1])

    def test_finance_note(self, oci_group):
        note = IFRS17Statements([oci_group], investment_return=[5, 5, 5, 5]) \
            .finance_income_and_expenses()
        np.testing.assert_allclose(note.loc["Net finance expenses from insurance contracts"],
                                   [0, -6, -5, -4])
        np.testing.assert_allclose(note.loc["Net financial result including OCI"],
                                   [5, -1, 0, 1])

    def test_reinsurance_oci(self):
        r = PAAReinsuranceHeld([240, 0], recoveries_incurred=[40, 42],
                               recoveries_received=[20, 36], finance_income=[0, 3],
                               finance_income_in_oci=[0, 1])
        profit = r.profit_or_loss()
        np.testing.assert_allclose(profit["reinsurance_finance_income"], [0, 2])
        np.testing.assert_allclose(profit["reinsurance_finance_income_oci"], [0, 1])


class TestStatementNotes:
    """Detailed presentation and supplementary information."""

    @pytest.fixture
    def statements(self, group, reinsurance):
        return IFRS17Statements([group], [reinsurance], opening_equity=500,
                                investment_return=[10, 12, 12, 13],
                                other_operating_expenses=20, tax_rate=0.25)

    def test_detailed_profit_or_loss(self, statements):
        detailed = statements.profit_or_loss(detailed=True)
        lines = ["Incurred claims and other insurance service expenses",
                 "Amortisation of insurance acquisition cash flows",
                 "Losses on onerous contracts and reversals",
                 "Adjustments to liabilities for incurred claims"]
        np.testing.assert_allclose(detailed.loc[lines].sum(),
                                   detailed.loc["Insurance service expenses"])
        # The totals are unchanged by showing the detail
        summary = statements.profit_or_loss()
        np.testing.assert_allclose(detailed.loc["Profit for the period"],
                                   summary.loc["Profit for the period"])
        assert len(detailed) == len(summary) + 4

    def test_insurance_service_expenses_note(self, statements):
        note = statements.insurance_service_expenses()
        np.testing.assert_allclose(note.loc["Insurance service expenses"],
                                   [-290, -267, -185, -260])
        np.testing.assert_allclose(note.loc["Amortisation of insurance acquisition cash flows"],
                                   -45)

    def test_supplementary_position(self, statements, group):
        """The IFRS 4 style components add up to the IFRS 17 carrying amount."""
        view = statements.supplementary_position()
        np.testing.assert_allclose(view.loc["Unearned premium"], [900, 600, 300, 0])
        np.testing.assert_allclose(view.loc["Deferred acquisition costs"], [-135, -90, -45, 0])
        # Premium received up front: nothing receivable
        np.testing.assert_allclose(view.loc["Premiums receivable"], 0, atol=1e-9)
        components = view.drop(index="Net insurance contract liabilities").sum()
        np.testing.assert_allclose(components, view.loc["Net insurance contract liabilities"])
        np.testing.assert_allclose(view.loc["Net insurance contract liabilities"],
                                   group.carrying_amount)

    def test_premiums_receivable_for_instalments(self):
        """Quarterly instalments: the unpaid premium of the year is receivable."""
        g = PAAGroup([300] * 4, expected_premium=1200, acquisition_cash_flows=[180, 0, 0, 0])
        view = g.supplementary_position()
        np.testing.assert_allclose(view["premiums_receivable"], [-900, -600, -300, 0])
        np.testing.assert_allclose(
            view["unearned_premium"] + view["premiums_receivable"]
            + view["deferred_acquisition_costs"],
            g.lrc,
        )


class TestGroupContracts:
    """Portfolio, cohort and profitability groups."""

    def test_grouping(self):
        from actuneo.ifrs17 import group_contracts
        policies = pd.DataFrame({
            "portfolio": ["Motor", "Motor", "Motor", "Fire", "Motor"],
            "start_date": ["2025-03-01", "2025-09-15", "2026-01-10", "2025-06-01", "2025-11-30"],
            "expected_combined_ratio": [0.70, 0.95, 1.10, 0.60, 1.05],
        })
        grouped = group_contracts(policies)
        assert grouped["cohort"].tolist() == ["2025", "2025", "2026", "2025", "2025"]
        assert grouped["profitability"].tolist() == [
            "no significant possibility of becoming onerous", "remaining", "onerous",
            "no significant possibility of becoming onerous", "onerous",
        ]
        assert grouped["group"].iloc[2] == "Motor | 2026 | onerous"
        assert grouped["group"].nunique() == 5
        assert "group" not in policies.columns

        quarterly = group_contracts(policies, cohort="Q")
        assert quarterly["cohort"].tolist() == ["2025Q1", "2025Q3", "2026Q1", "2025Q2", "2025Q4"]
        strict = group_contracts(policies, no_significant_possibility_below=0.65)
        assert strict["profitability"].iloc[0] == "remaining"

        with pytest.raises(ValueError):
            group_contracts(policies, cohort="W")
        with pytest.raises(ValueError):
            group_contracts(policies, onerous_above=0.5)


class TestDisclosures:
    """Claims development, maturity analysis and sensitivities."""

    @pytest.fixture
    def paid(self):
        return Triangle([[100, 150, 165], [110, 176, np.nan], [120, np.nan, np.nan]],
                        origin=[2023, 2024, 2025], development=[0, 1, 2])

    def test_claims_development_table_by_hand(self, paid):
        from actuneo.ifrs17 import claims_development_table
        table = claims_development_table(paid, risk_adjustment=10, earlier_years_liability=5)
        f1, f2 = 326 / 210, 1.1
        # Estimates rebuilt with today's development factors
        np.testing.assert_allclose(table.loc["At end of accident year", [2023, 2024, 2025]],
                                   [100 * f1 * f2, 110 * f1 * f2, 120 * f1 * f2])
        np.testing.assert_allclose(table.loc["One year later", [2023, 2024]],
                                   [150 * f2, 176 * f2])
        assert table.loc["Two years later", 2023] == 165
        assert np.isnan(table.loc["Two years later", 2024])

        cl = ChainLadder(paid)
        np.testing.assert_allclose(
            table.loc["Current estimate of cumulative claims", [2023, 2024, 2025]], cl.ultimate
        )
        assert table.loc["Liabilities for the accident years shown", "Total"] == pytest.approx(
            cl.reserve()
        )
        assert table.loc["Liabilities for incurred claims", "Total"] == pytest.approx(
            cl.reserve() + 5 + 10
        )

    def test_claims_development_ties_to_the_discounted_liability(self):
        from actuneo.ifrs17 import claims_development_table
        raa = load_raa()
        mack = MackChainLadder(raa, est_sigma="mack")
        lic = LiabilityForIncurredClaims.from_reserving(mack, 0.08, confidence_level=0.75)
        table = claims_development_table(raa, discount_rate=0.08,
                                         risk_adjustment=lic.risk_adjustment)
        assert table.loc["Effect of discounting", "Total"] == pytest.approx(-lic.discounting)
        assert table.loc["Liabilities for incurred claims", "Total"] == pytest.approx(lic.total)
        assert table.shape == (17, 11)

    def test_recorded_estimates(self, paid):
        from actuneo.ifrs17 import claims_development_table
        recorded = pd.DataFrame([[170, 168, 165], [190, 195, np.nan], [200, np.nan, np.nan]],
                                index=[2023, 2024, 2025], columns=[0, 1, 2])
        table = claims_development_table(paid, estimates=recorded, effect_of_discounting=8)
        assert table.loc["At end of accident year", 2023] == 170
        np.testing.assert_allclose(
            table.loc["Current estimate of cumulative claims", [2023, 2024, 2025]],
            [165, 195, 200],
        )
        np.testing.assert_allclose(
            table.loc["Liabilities for the accident years shown", [2023, 2024, 2025]],
            [0, 19, 80],
        )
        assert table.loc["Liabilities for incurred claims", "Total"] == 99 - 8

    def test_maturity_analysis(self):
        from actuneo.ifrs17 import maturity_analysis
        flows = [100, 80, 60, 40, 20, 10, 5]
        undiscounted = maturity_analysis(flows)
        assert undiscounted["1 year or less"] == 100
        assert undiscounted["4-5 years"] == 20
        assert undiscounted["More than 5 years"] == 15
        assert undiscounted["Total"] == 315

        discounted = maturity_analysis(flows, 0.10)
        assert discounted["1 year or less"] == pytest.approx(100 * 1.1 ** -0.5)
        assert discounted["Total"] == pytest.approx(
            LiabilityForIncurredClaims(flows, 0.10).present_value
        )
        # Quarterly periods: the first four fall within one year
        quarterly = maturity_analysis(flows, periods_per_year=4)
        assert quarterly["1 year or less"] == 280 and quarterly["1-2 years"] == 35

    def test_sensitivity(self):
        from actuneo.ifrs17 import lic_sensitivity
        table = lic_sensitivity([100, 50], discount_rate=0.10, risk_adjustment=12,
                                tax_rate=0.25)
        base = LiabilityForIncurredClaims([100, 50], 0.10, risk_adjustment=12).total
        assert table.loc["Base", "liability"] == pytest.approx(base)
        assert table.loc["Claims +5%", "liability"] == pytest.approx(1.05 * base)
        assert table.loc["Claims +5%", "profit_before_tax"] == pytest.approx(-0.05 * base)
        assert table.loc["Claims +5%", "equity"] == pytest.approx(-0.05 * base * 0.75)
        higher = 100 * 1.11 ** -0.5 + 50 * 1.11 ** -1.5 + 12
        assert table.loc["Discount rate +1.0%", "liability"] == pytest.approx(higher)
        assert table.loc["Discount rate +1.0%", "profit_before_tax"] > 0

        curve = YieldCurve([1, 5], [0.10, 0.10])
        assert len(lic_sensitivity([100, 50], curve)) == 3


class TestSignificantFinancingComponent:
    """Time value of money in the liability for remaining coverage (IFRS 17.56)."""

    def test_two_year_cover_paid_up_front(self):
        g = PAAGroup([1000, 0], lrc_discount_rate=0.05)
        v = 1 / 1.05
        level = 1000 / (v + v ** 2)
        np.testing.assert_allclose(g.insurance_revenue, level)
        np.testing.assert_allclose(g.lrc_interest, [50, (1050 - level) * 0.05])
        np.testing.assert_allclose(g.lrc, [1050 - level, 0], atol=1e-9)
        # Revenue is the premium plus the interest accreted
        assert g.insurance_revenue.sum() == pytest.approx(1000 + g.lrc_interest.sum())

    def test_interest_is_a_finance_expense(self):
        g = PAAGroup([1000, 0], lrc_discount_rate=0.05)
        profit = g.profit_or_loss()
        np.testing.assert_allclose(profit["insurance_finance_expenses"], -g.lrc_interest)
        # Over the whole cover the extra revenue and the interest expense cancel
        total = profit["insurance_service_result"].sum() + \
            profit["insurance_finance_expenses"].sum()
        assert total == pytest.approx(1000)
        change = np.diff(np.concatenate(([0.0], g.carrying_amount)))
        result = profit["insurance_service_result"] + profit["insurance_finance_expenses"]
        np.testing.assert_allclose(change, g.cash_flows()["net_cash_flow"] - result)
        table = g.reconciliation(1)
        assert table.loc["Net finance expenses from insurance contracts",
                         "lrc_excluding_loss_component"] == pytest.approx(50)

    def test_premium_in_arrears_earns_finance_income(self):
        """Cover given before the premium is paid: the insurer is the lender."""
        g = PAAGroup([0, 0, 1000], lrc_discount_rate=0.05)
        # Interest is earned while cover has been given but the premium is still unpaid
        assert g.lrc_interest[1] < 0 and g.lrc_interest.sum() < 0
        assert g.lrc[-1] == pytest.approx(0, abs=1e-9)
        assert g.insurance_revenue.sum() < 1000

    def test_quarterly_periods_and_zero_rate(self):
        quarterly = PAAGroup([1200, 0, 0, 0], lrc_discount_rate=0.08, periods_per_year=4)
        rate = 1.08 ** 0.25 - 1
        assert quarterly.lrc_interest[0] == pytest.approx(1200 * rate)
        assert quarterly.lrc[-1] == pytest.approx(0, abs=1e-9)
        none = PAAGroup([1200, 0, 0, 0], lrc_discount_rate=0.0)
        np.testing.assert_allclose(none.insurance_revenue, 300)

    def test_with_acquisition_cash_flows_and_statements(self):
        g = PAAGroup([1000, 0], acquisition_cash_flows=[100, 0], lrc_discount_rate=0.05)
        assert g.lrc[-1] == pytest.approx(0, abs=1e-9)
        assert g.amortisation.sum() == pytest.approx(100)
        assert IFRS17Statements([g], opening_equity=200).balance_check() == pytest.approx(0)

    def test_invalid(self):
        with pytest.raises(ValueError, match="revenue_pattern"):
            PAAGroup([1000, 0], insurance_revenue=[500, 500], lrc_discount_rate=0.05)
        with pytest.raises(ValueError, match="whole of the remaining cover"):
            PAAGroup([1000, 0], revenue_pattern=[0.25, 0.25], lrc_discount_rate=0.05)
