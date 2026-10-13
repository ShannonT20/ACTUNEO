"""
Tests of actuneo.ai.ask: reading a question, choosing the calculation, and the
worked solution.

Every solver's answer is compared with a direct call to the library. The
language model is replaced by a stand-in.
"""

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

from actuneo import ai, zimbabwe
from actuneo.ai import knowledge
from actuneo.ai.assistant import (ACTUARY_PROMPT, extract_arguments, normalise, rank_solvers,
                                  read_data, split_parts)
from actuneo.ai.benchmark import arguments_match
from actuneo.ai.solvers import SOLVERS
from actuneo.finance import Bond, Cashflows, Loan
from actuneo.ifrs17 import PAAGroup
from actuneo.life import LifeAssurance
from actuneo.loss_reserving import ChainLadder, MackChainLadder, load_genins, load_raa
from actuneo.mortality import MortalityTable, load_am92
from actuneo.pensions import DCProjection, pension_annuity

assistant = importlib.import_module("actuneo.ai.assistant")

# (question, calculation, arguments) answered correctly by the rules alone
QUESTIONS = [
    ("What is the life expectancy of a 45 year old woman?", "life_expectancy", {"age": 45, "sex": "F"}),
    ("How long is a man aged 70 expected to live?", "life_expectancy", {"age": 70, "sex": "M"}),
    ("What is the probability that a 30 year old survives 20 years?", "survival_probability", {"age": 30, "years": 20}),
    ("Value a pension of 6000 a year from age 65 at 5% interest.", "life_annuity", {"age": 65, "annual_amount": 6000, "interest_rate": 0.05, "sex": "M"}),
    ("Premium for a 20-year term assurance of 100000 for a 35 year old at 6% interest", "term_assurance_premium", {"age": 35, "term": 20, "sum_assured": 100000, "interest_rate": 0.06}),
    ("Estimate the IBNR for the RAA triangle with the chain ladder method.", "chain_ladder_reserve", {"triangle": "RAA"}),
    ("What is the Mack standard error for the GenIns triangle?", "mack_standard_error", {"triangle": "GenIns"}),
    ("What is the monthly repayment on a loan of 50000 over 10 years at 8% interest?", "loan_repayment", {"amount": 50000, "interest_rate": 0.08, "years": 10}),
    ("Price a 10-year bond with a 6% coupon at a yield of 7%.", "bond_price", {"coupon_rate": 0.06, "years": 10, "yield_rate": 0.07}),
    ("What is the present value of 1000 due in 5 years at 10% interest?", "present_value", {"amount": 1000, "years": 5, "interest_rate": 0.10}),
    ("Convert 1,000,000 ZWL to ZiG.", "zwl_to_zig", {"amount": 1000000}),
    ("I am 30 and retire at 60 on a salary of 12000, contributing 10% with a 7% investment return. How big is my retirement fund?", "retirement_fund", {"age": 30, "retirement_age": 60, "salary": 12000, "contribution_rate": 0.10, "investment_return": 0.07}),
    ("If I invest 5,000 today at 9%, what will it grow to in 12 years?", "future_value", {"amount": 5000, "years": 12, "interest_rate": 0.09}),
    ("What is the effective annual rate of 12% convertible monthly?", "effective_rate", {"nominal_rate": 0.12, "periods": 12}),
    ("Present value of payments of 1000 at the end of each year for 10 years at 6%", "annuity_certain", {"payment": 1000, "years": 10, "interest_rate": 0.06, "timing": "arrears"}),
    # natural wording
    ("My client is a 52-year-old lady. How many more years can she expect?", "life_expectancy", {"age": 52, "sex": "F"}),
    ("Chance a 50 year old lives another 10 years?", "survival_probability", {"age": 50, "years": 10}),
    ("What does it cost to give a man of 60 an income of 12,000 a year for the rest of his life, at 4%?", "life_annuity", {"age": 60, "annual_amount": 12000, "interest_rate": 0.04, "sex": "M"}),
    ("How much a year for 250,000 of life cover over 25 years for someone aged 40? Use 5% interest.", "term_assurance_premium", {"age": 40, "term": 25, "sum_assured": 250000, "interest_rate": 0.05}),
    ("What's the IBNR on GenIns?", "chain_ladder_reserve", {"triangle": "GenIns"}),
    ("How uncertain is the chain ladder reserve for RAA?", "mack_standard_error", {"triangle": "RAA"}),
    ("Borrowing 120k over 15 years at eight and a half percent. What do I pay each month?", "loan_repayment", {"amount": 120000, "interest_rate": 0.085, "years": 15}),
    ("A 15-year bond pays a 9% coupon and yields are 11%. What should it trade at?", "bond_price", {"coupon_rate": 0.09, "years": 15, "yield_rate": 0.11}),
    ("What is 25,000 payable in 12 years worth today at 7.5%?", "present_value", {"amount": 25000, "years": 12, "interest_rate": 0.075}),
    ("How much is ZWL 5 million in ZiG?", "zwl_to_zig", {"amount": 5000000}),
    ("A woman aged 60 wants 12,000 a year for life. What does that cost at 4%?", "life_annuity", {"age": 60, "annual_amount": 12000, "interest_rate": 0.04, "sex": "F"}),
    # working needed before the calculation
    ("If I am 45 now, what are the odds I am still alive at 70?", "survival_probability", {"age": 45, "years": 25}),
    ("A widow aged 63 will receive 750 a month for life. What is that worth at 6%?", "life_annuity", {"age": 63, "annual_amount": 9000, "interest_rate": 0.06, "sex": "F"}),
    ("I am 28, earn 18,000 a year and retire at 60. I put in 7.5% and my employer adds 10%. Returns are 8%. How big is my pot?", "retirement_fund", {"age": 28, "retirement_age": 60, "salary": 18000, "contribution_rate": 0.175, "investment_return": 0.08}),
    ("A bond issued 5 years ago with a 20-year term pays 8%. Yields are 6%. What is the price now?", "bond_price", {"coupon_rate": 0.08, "years": 15, "yield_rate": 0.06}),
    ("Our Harare branch lent 80,000 to a 40 year old teacher over 20 years at 12%. Monthly instalment?", "loan_repayment", {"amount": 80000, "interest_rate": 0.12, "years": 20}),
    ("The RAA triangle has 10 accident years. What reserve does chain ladder give?", "chain_ladder_reserve", {"triangle": "RAA"}),
    # investments and IFRS 17
    ("What is the NPV at 8% of cash flows of -1000, 300, 400, 500?", "net_present_value", {"interest_rate": 0.08}),
    ("IFRS 17 PAA: premium of 1200 for a 12-month policy with acquisition costs of 180. What is the LRC after 3 months?", "paa_liability", {"premium": 1200, "elapsed_months": 3, "coverage_months": 12, "acquisition_costs": 180}),
    ("Under the premium allocation approach a premium of 2400 was received for a 6-month policy. What is the liability for remaining coverage after 2 months?", "paa_liability", {"premium": 2400, "elapsed_months": 2, "coverage_months": 6, "acquisition_costs": 0}),
]


@pytest.fixture(autouse=True)
def no_model(monkeypatch, tmp_path):
    """No language model unless a test supplies one."""
    monkeypatch.setenv("ACTUNEO_AI_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(assistant, "_backend", lambda use_model: None)
    warnings.simplefilter("ignore")


class TestReadingQuestions:
    @pytest.mark.parametrize("question, solver, arguments", QUESTIONS)
    def test_rules_alone_read_the_question(self, question, solver, arguments):
        answer = ai.ask(question)
        assert answer.solved, answer.text
        assert answer.solver == solver
        assert arguments_match(arguments, answer.arguments), answer.arguments
        assert not answer.used_model
        assert set(answer.arguments) == {p.name for p in SOLVERS[solver].params}

    def test_numbers_are_put_in_one_form(self):
        assert normalise("a loan of 1,250,000 at 8.5%") == "a loan of 1250000 at 8.5%"
        assert normalise("120k over 15 years") == "120000 over 15 years"
        assert normalise("ZWL 5 million and 2.5m") == "ZWL 5000000 and 2500000"
        assert normalise("at eight and a half percent") == "at 8.5%"
        assert normalise("twenty five per cent") == "25%"
        assert normalise("seven point five percent") == "7.5%"
        assert normalise("six and three quarters percent") == "6.75%"
        assert normalise("12 percent and 3 pct") == "12% and 3%"
        assert normalise("5 months") == "5 months"

    def test_working_that_was_done_is_reported(self):
        survive = ai.ask("If I am 45 now, what are the odds I am still alive at 70?")
        assert "age 70 - age 45 = 25 years" in " ".join(survive.notes)
        monthly = ai.ask("A widow aged 63 will receive 750 a month for life, worth at 6%?")
        assert "750 a month x 12 = 9000 a year" in " ".join(monthly.notes)
        pot = ai.ask("I am 28, earn 18,000 a year and retire at 60. I put in 7.5% and my "
                     "employer adds 10%. Returns are 8%. How big is my pot?")
        assert "7.5% from the member + 10% from the employer = 17.5%" in " ".join(pot.notes)
        bond = ai.ask("A bond issued 5 years ago with a 20-year term pays 8%. Yields are 6%.")
        assert "= 15 years left" in " ".join(bond.notes)

    def test_names_that_contain_numbers_are_not_read_as_quantities(self):
        found, _, missing = extract_arguments(SOLVERS["paa_liability"],
                                              "what does actuneo do for ifrs 17?")
        assert found == {} and [p.name for p in missing] == ["premium", "elapsed_months"]
        assert not assistant.has_quantities("IAS 19 and IFRS 17 under SI 67")
        assert assistant.has_quantities("a loan of 500 under IFRS 17")

    def test_an_age_is_not_mistaken_for_a_term(self):
        found, _, missing = extract_arguments(
            SOLVERS["loan_repayment"],
            normalise("lent 80,000 to a 40 year old teacher over 20 years at 12%"))
        assert found == {"amount": 80000, "interest_rate": 0.12, "years": 20} and not missing

    def test_missing_information_is_asked_for_not_guessed(self):
        loan = ai.ask("What is the monthly repayment on my loan?")
        assert loan.kind == "need_more" and not loan.solved and loan.value is None
        assert "the amount borrowed, the interest rate, the term of the loan" in loan.text
        pension = ai.ask("Value my pension.")
        assert pension.kind == "need_more" and "the age" in pension.text
        trap = ai.ask("What is the life expectancy of a 10-year bond?")
        assert trap.kind == "need_more" and "the age" in trap.text
        partial = ai.ask("Repayment on a loan of 5000 at 10%?")
        assert partial.arguments == {"amount": 5000, "interest_rate": 0.10}
        assert partial.text.count("the term of the loan in years") == 1

    def test_values_outside_a_sensible_range_are_refused(self):
        old = ai.ask("What is the life expectancy of a 400 year old man?")
        assert old.kind == "need_more" and "outside the range" in old.text
        off_table = ai.ask("What is the probability that a 5 year old survives 10 years?")
        assert off_table.kind == "need_more" and "covers ages 20 to 100" in off_table.text
        assert "Name another basis" in off_table.text
        late = ai.ask("PAA premium of 1200 for a 12-month policy, LRC after 15 months?")
        assert late.kind == "need_more" and "cannot exceed the coverage period" in late.text

    def test_greetings_and_unrelated_questions(self):
        hello = ai.ask("Good morning")
        assert hello.kind == "greeting" and "Chain ladder reserve" in hello.text
        assert ai.ask("").kind == "unknown"
        for question in ["What is the weather like on Mars today, xyzzy?",
                         "What is the probability that it rains in Bulawayo tomorrow?"]:
            answer = ai.ask(question)
            assert not answer.solved and answer.kind in ("unknown", "reference")

    def test_ranking(self):
        ranked = rank_solvers("how uncertain is the chain ladder reserve for raa?")
        assert ranked[0][1].name == "mack_standard_error" and ranked[1][1].name == "chain_ladder_reserve"
        assert rank_solvers("hello there") == []
        assert len(ai.topics()) == len(SOLVERS) == 16
        boosted = rank_solvers("what is the reserve?", ("triangle",))
        assert boosted[0][1].name == "chain_ladder_reserve" and boosted[0][0] == 4


class TestSolutionsAgreeWithTheLibrary:
    def value(self, question):
        answer = ai.ask(question)
        assert answer.solved, answer.text
        return answer.value

    def test_compound_interest(self):
        assert self.value("Present value of 1000 due in 5 years at 10%") == pytest.approx(1000 / 1.1 ** 5)
        assert self.value("If I invest 5000 at 9% what will it grow to in 12 years?") == \
            pytest.approx(5000 * 1.09 ** 12)
        assert self.value("Effective annual rate of 12% convertible monthly") == \
            pytest.approx(1.01 ** 12 - 1)
        assert self.value("Effective annual rate of 8% nominal compounded quarterly") == \
            pytest.approx(1.02 ** 4 - 1)
        arrears = self.value("Present value of payments of 1000 at the end of each year for 10 years at 6%")
        advance = self.value("Present value of payments of 1000 at the start of each year for 10 years at 6%")
        assert arrears == pytest.approx(1000 * (1 - 1.06 ** -10) / 0.06)
        assert advance == pytest.approx(arrears * 1.06)

    def test_loan_and_bond(self):
        assert self.value("Monthly repayment on a loan of 50000 over 10 years at 8%") == \
            pytest.approx(Loan(50000, 0.08, 10, payments_per_year=12).payment)
        assert self.value("Price a 10-year bond with a 6% coupon at a yield of 7%") == \
            pytest.approx(Bond(0.06, 10, frequency=2).price(0.07))

    def test_zimbabwe_tables_are_the_default(self):
        assured = MortalityTable.from_zimbabwe_2023("male_assured_lives")
        answer = ai.ask("Life expectancy of a 45 year old man")
        assert answer.arguments["basis"] == "zimbabwe"
        assert answer.value == pytest.approx(assured.life_expectancy(45))
        assert "Zimbabwe 2023 male assured lives" in answer.explain()
        assert "South African" not in answer.explain()
        woman = ai.ask("Life expectancy of a 45 year old woman")
        assert woman.value == pytest.approx(assured.life_expectancy(45))
        assert "no female assured lives table is published" in woman.explain()
        assert self.value("Probability that a 30 year old survives 20 years") == \
            pytest.approx(assured.npx(30, 20))
        assert self.value("Premium for a 20-year term assurance of 100000 for a 35 year old at 6%") == \
            pytest.approx(100000 * LifeAssurance(assured, interest_rate=0.06)
                          .net_annual_premium(35, 20, "term"))

    def test_zimbabwe_pensions(self):
        pensioners = MortalityTable.from_zimbabwe_2023("post_retirement_pensions_female")
        old = ai.ask("Value a pension of 6000 a year from age 80 at 5% interest for a woman")
        assert old.value == pytest.approx(6000 * pension_annuity(pensioners, 80, 0.05))
        assert "Zimbabwe 2023 post-retirement pensions, females" in old.explain()
        # Below 76 the published assured lives and pensioner rates are joined, and it says so
        assured = MortalityTable.from_zimbabwe_2023("male_assured_lives")
        men = MortalityTable.from_zimbabwe_2023("post_retirement_pensions_male")
        joined = MortalityTable(list(range(20, 76)) + list(men.ages),
                                list(assured.qx(list(range(20, 76)))) + list(men.qx_values))
        young = ai.ask("Value a pension of 6000 a year from age 65 at 5% interest.")
        assert young.value == pytest.approx(6000 * pension_annuity(joined, 65, 0.05))
        assert "This join is ACTUNEO's, not a published table" in young.explain()

    def test_other_bases_only_when_named(self):
        males = MortalityTable.from_south_africa("saiml98")
        assert self.value("Value a pension of 6000 a year from age 65 at 5%, South African tables") == \
            pytest.approx(6000 * pension_annuity(males, 65, 0.05))
        assert self.value("Life expectancy of a 30 year old man on SA85-90") == pytest.approx(
            MortalityTable.from_south_africa("sa8590").life_expectancy(30))
        exam = ai.ask("What is the life expectancy of a 45 year old man on AM92?")
        assert exam.arguments["basis"] == "am92"
        assert exam.value == pytest.approx(load_am92().ultimate.life_expectancy(45))
        assert "not a basis for African lives" in exam.explain()

    def test_investment_appraisal(self):
        flows = Cashflows([0, 1, 2, 3], [-1000, 300, 400, 500])
        answer = ai.ask("What is the NPV at 8% of cash flows of -1000, 300, 400, 500?")
        assert answer.value == pytest.approx(flows.net_present_value(0.08))
        text = answer.explain()
        assert f"{flows.internal_rate_of_return():.2%}" in text and "v^t" in text
        only_rate = ai.ask("What is the internal rate of return of cash flows -1000, 300, 400, 500?")
        assert only_rate.value == pytest.approx(flows.internal_rate_of_return())
        none = ai.ask("What is the NPV at 8%?")
        assert none.kind == "need_more" and "I need the cash flows" in none.text

    def test_ifrs17_premium_allocation(self):
        group = PAAGroup(premiums_received=[1200, 0, 0, 0], acquisition_cash_flows=[180, 0, 0, 0],
                         periods=["Q1", "Q2", "Q3", "Q4"])
        rollforward = group.lrc_rollforward()
        for quarter, months in zip(["Q1", "Q2", "Q3", "Q4"], [3, 6, 9, 12]):
            answer = ai.ask(f"IFRS 17 PAA: premium of 1200 for a 12-month policy with "
                            f"acquisition costs of 180. What is the LRC after {months} months?")
            assert answer.value == pytest.approx(rollforward.loc[quarter, "closing"])
        text = answer.explain()
        assert "IFRS 17.B126(a)" in text and "liability for incurred claims is separate" in text

    def test_reserving(self):
        assert self.value("Chain ladder IBNR for the RAA triangle") == \
            pytest.approx(ChainLadder(load_raa()).total_ibnr)
        working = ai.ask("Chain ladder IBNR for the RAA triangle").explain()
        assert "origin years  10" in working and "For 1990: 2,063 / 0.112105 = 18,402" in working
        assert "For Total" not in working
        assert "The most recent origin year carries 31% of the reserve" in working
        mack = MackChainLadder(load_genins(), est_sigma="mack")
        answer = ai.ask("Mack standard error for GenIns")
        assert answer.value == pytest.approx(mack.total_mack_se)
        # The two sources of error combine as the working says they do
        assert mack.total_mack_se ** 2 == pytest.approx(
            mack.total_process_risk ** 2 + mack.total_parameter_risk ** 2)

    def test_pensions_and_currency(self):
        assert self.value("I am 30 and retire at 60 on a salary of 12000, contributing 10% "
                          "with a 7% return. How big is my pot?") == pytest.approx(
            DCProjection(30, 60, 12000, member_rate=0.10, investment_return=0.07).fund_at_retirement)
        rising = ai.ask("I am 30, retire at 60, salary of 12000 with salary increases of 4%, "
                        "contributing 10%, return 7%. Retirement fund?")
        assert rising.arguments["salary_increase"] == pytest.approx(0.04)
        assert rising.value == pytest.approx(DCProjection(
            30, 60, 12000, member_rate=0.10, investment_return=0.07,
            salary_increase=0.04).fund_at_retirement)
        assert self.value("Convert 1,000,000 ZWL to ZiG") == pytest.approx(
            zimbabwe.zwl_to_zwg(1_000_000))

    @pytest.mark.parametrize("name", sorted(SOLVERS))
    def test_every_example_is_solved_and_set_out(self, name):
        solver = SOLVERS[name]
        answer = ai.ask(solver.example)
        assert answer.solved and answer.solver == name
        solution = answer.solution
        assert solution.given and len(solution.steps) >= 2 and solution.assumptions
        assert all(step.title and step.lines for step in solution.steps)
        text = answer.explain()
        for heading in ("Question:", "How the question was read", "Given", "Working", "Answer",
                        "Assumptions"):
            assert heading in text
        markdown = answer.to_markdown()
        assert markdown == answer._repr_markdown_() and "**Working**" in markdown
        assert "**Answer:**" in markdown and repr(answer) == text
        assert str(solution) == solution.explain()
        tool = solver.tool()
        assert tool["name"] == name and set(tool["parameters"]["required"]) <= set(
            tool["parameters"]["properties"])

    def test_the_working_can_be_followed(self):
        text = ai.ask("What is the present value of 1000 due in 5 years at 10% interest?").explain()
        for line in ["v = 1 / (1 + i)", "= 1 / 1.1", "= 0.909091", "= 0.909091^5", "= 0.620921",
                     "PV = S x v^n", "= 1,000.00 x 0.620921", "= 620.92",
                     "The present value is 620.92."]:
            assert line in text
        loan = ai.ask("Monthly repayment on a loan of 50000 over 10 years at 8%").explain()
        assert "j = (1 + i)^(1/12) - 1" in loan and "= 599.29" in loan
        assert "nominal rate convertible monthly" in loan and "606.64" in loan
        life = ai.ask("Probability that a 30 year old survives 20 years").explain()
        assert "out of 100,000 at age 20" in life and "n_p_x = l_(x+n) / l_x" in life


class TestPartsAndOwnData:
    def test_splitting(self):
        assert split_parts("What is the present value of 1000 due in 5 years at 10%?") == (
            "", ["What is the present value of 1000 due in 5 years at 10%?"])
        preamble, parts = split_parts("A man is aged 40. (a) What is his life expectancy? "
                                      "(b) Chance he survives 20 years?")
        assert preamble == "A man is aged 40."
        assert parts == ["What is his life expectancy?", "Chance he survives 20 years?"]
        preamble, parts = split_parts("A loan of 5000 runs 10 years. What is the repayment at "
                                      "8%? What is it at 10%? Use monthly instalments.")
        assert preamble == "A loan of 5000 runs 10 years." and len(parts) == 2
        assert parts[1] == "What is it at 10%? Use monthly instalments."
        assert len(split_parts("Find the present value of 100 in 5 years at 6% and what is "
                               "the future value of 100 in 5 years at 6%?")[1]) == 2
        # One question followed by a statement stays whole
        assert len(split_parts("How much for life cover for someone aged 40? Use 5%.")[1]) == 1
        assert len(split_parts("1. Present value of 10 in 2 years at 5%? 2. Convert 9 ZWL to ZiG")[1]) == 2

    def test_each_part_is_answered_with_what_the_opening_gave(self):
        answer = ai.ask("A man is aged 40. (a) What is his life expectancy? (b) What is the "
                        "probability he survives 20 years? (c) What is the premium for a 10-year "
                        "term assurance of 50000 at 6%?")
        assert answer.kind == "parts" and answer.solved and answer.value is None
        assert [part.solver for part in answer.parts] == [
            "life_expectancy", "survival_probability", "term_assurance_premium"]
        assert all(part.arguments["age"] == 40 for part in answer.parts)
        assert "Taken from the opening of the question: the age." in answer.parts[1].notes
        table = MortalityTable.from_zimbabwe_2023("male_assured_lives")
        assert answer.parts[1].value == pytest.approx(table.npx(40, 20))
        text = answer.explain()
        assert "----- Part 1 -----" in text and "----- Part 3 -----" in text
        assert "**Part 2**" in answer.to_markdown()

    def test_two_unrelated_questions_in_one(self):
        answer = ai.ask("What is the present value of 1000 due in 5 years at 10% and what is "
                        "the monthly repayment on a loan of 50000 over 10 years at 8%?")
        assert [part.solver for part in answer.parts] == ["present_value", "loan_repayment"]
        assert answer.parts[0].value == pytest.approx(1000 / 1.1 ** 5)

    def test_a_part_that_cannot_be_answered_says_why(self):
        answer = ai.ask("(a) Present value of 1000 due in 5 years at 10%? (b) Value my pension.")
        assert answer.parts[0].solved and answer.parts[1].kind == "need_more"
        assert not answer.solved

    def test_one_answer_for_each_triangle_named(self):
        answer = ai.ask("Chain ladder reserve for RAA and GenIns")
        assert answer.kind == "parts"
        assert [part.arguments["triangle"] for part in answer.parts] == ["RAA", "GenIns"]
        assert answer.parts[1].value == pytest.approx(ChainLadder(load_genins()).total_ibnr)

    def test_own_triangle(self, tmp_path):
        rows = [[100, 150, 170, 180], [110, 168, 190], [120, 180], [130]]
        from actuneo.loss_reserving import Triangle
        grid = pd.DataFrame(rows, index=[2021, 2022, 2023, 2024])
        expected = ChainLadder(Triangle(grid)).total_ibnr
        for data in (rows, grid, {"triangle": grid}, Triangle(grid)):
            answer = ai.ask("What is the chain ladder reserve for my triangle?", data=data)
            assert answer.arguments == {"triangle": "own"}
            assert answer.value == pytest.approx(expected)
            assert "The triangle is the data you passed." in answer.notes
        path = tmp_path / "triangle.csv"
        grid.rename_axis("origin").to_csv(path)
        from_file = ai.ask("IBNR reserve?", data=str(path))
        assert from_file.solver == "chain_ladder_reserve"
        assert from_file.value == pytest.approx(expected)
        assert ai.ask("How uncertain is that reserve?", data=rows).solver == "mack_standard_error"
        # A named triangle still wins over the data passed
        assert ai.ask("Chain ladder reserve for RAA", data=rows).arguments == {"triangle": "RAA"}
        missing = ai.ask("Chain ladder reserve for my triangle")
        assert missing.kind == "need_more" and "did not pass one" in missing.text

    def test_own_mortality_table(self):
        ages = np.arange(20, 111)
        rates = np.minimum(0.0005 * 1.09 ** (ages - 20), 1.0)
        rates[-1] = 1.0
        frame = pd.DataFrame({"Age": ages, "qx": rates})
        table = MortalityTable(ages, rates)
        answer = ai.ask("What is the life expectancy at age 60?", data=frame)
        assert answer.arguments["basis"] == "own"
        assert answer.value == pytest.approx(table.life_expectancy(60))
        assert "your own table" in answer.explain()
        pension = ai.ask("Value a pension of 1000 a year from age 65 at 5%", data=table)
        assert pension.value == pytest.approx(1000 * pension_annuity(table, 65, 0.05))
        # Naming a basis overrides the table passed
        named = ai.ask("Life expectancy at age 60 on the Zimbabwe tables", data=frame)
        assert named.arguments["basis"] == "zimbabwe"

    def test_own_cash_flows(self):
        flows = [-5000, 1500, 1500, 1500, 1500]
        expected = Cashflows(range(5), flows)
        assert ai.ask("NPV at 10%?", data=flows).value == pytest.approx(
            expected.net_present_value(0.10))
        assert ai.ask("What is the IRR of this project?", data=pd.Series(flows)).value == \
            pytest.approx(expected.internal_rate_of_return())

    def test_data_is_recognised(self, tmp_path):
        assert list(read_data(None)) == []
        assert list(read_data([1, 2, 3])) == ["cashflows"]
        assert list(read_data([[1, 2], [3]])) == ["triangle"]
        assert list(read_data(pd.DataFrame({"age": [30, 31], "qx": [0.1, 1.0]}))) == ["table"]
        assert list(read_data(pd.DataFrame({"flow": [-10, 6, 6]}))) == ["cashflows"]
        both = read_data({"triangle": [[1, 2], [3]], "cashflows": [1, 2]})
        assert set(both) == {"triangle", "cashflows"}
        bad = ai.ask("Chain ladder reserve?", data=str(tmp_path / "absent.csv"))
        assert bad.kind == "need_more" and "was not found" in bad.text


class TestKnowledge:
    def test_index_is_built_from_what_already_exists(self):
        passages = knowledge.corpus()
        sources = {passage.source.split(".")[0] if passage.source.startswith("actuneo")
                   else passage.source for passage in passages}
        assert {"README.md", "docs/status.rst", "docs/pensions.rst", "actuneo"} <= sources
        assert len(passages) > 300
        assert all(len(passage.text.split()) <= knowledge.MAX_WORDS + 1 for passage in passages)
        assert not any(".ai." in passage.source for passage in passages)

    def test_stored_documentation_is_up_to_date(self):
        root = knowledge.Path(knowledge.__file__).resolve().parents[2]
        if not (root / "docs" / "status.rst").exists():
            pytest.skip("not a source checkout")
        fresh = []
        for name in knowledge.DOCUMENTS:
            fresh.extend(knowledge.passages_from_document(root / name, name))
        stored = knowledge._documentation()
        assert [(p.title, p.text) for p in stored] == [(p.title, p.text) for p in fresh], \
            "run actuneo.ai.knowledge.build_corpus_file() after changing the documentation"

    @pytest.mark.parametrize("question, expected", [
        ("How do I load a Zimbabwe mortality table?", "zimbabwe"),
        ("What does ACTUNEO do for IFRS 17 PAA?", "actuneo.ifrs17"),
        ("how do I compute bond duration", "duration"),
        ("What are the limitations of the pensions module?", "pensions"),
        ("how do I run a chain ladder on a triangle", "loss_reserving"),
    ])
    def test_retrieval_finds_the_relevant_passages(self, question, expected):
        found = knowledge.retrieve(question, limit=4)
        assert found and found[0][0] >= found[-1][0]
        assert any(expected in (passage.source + passage.title).lower() for _, passage in found)

    def test_nothing_relevant(self):
        assert knowledge.retrieve("xyzzy plugh") == []

    def test_without_a_model_the_passages_are_returned(self):
        answer = ai.ask("What does ACTUNEO do for IFRS 17?")
        assert answer.kind == "reference" and not answer.used_model and not answer.solved
        assert "No language model is installed" in answer.text and answer.sources
        assert "Passages of ACTUNEO's documentation it drew on" in answer.explain()
        about = ai.ask("What is ACTUNEO?")
        assert about.kind == "reference" and about.sources[0].source == "README.md"


class FakeBackend:
    """A stand-in for the language model that returns a scripted choice and reply."""

    def __init__(self, name=None, arguments=None, reply="An actuary's answer."):
        self.name, self.arguments, self.reply = name, arguments or {}, reply
        self.calls, self.prompts = [], []

    def call_tools(self, prompt, tools, system=None, allow_none=True):
        self.calls.append([tool["name"] for tool in tools])
        return ai.ToolCall(self.name, dict(self.arguments))

    def generate(self, prompt, system=None, max_tokens=None, temperature=0.2):
        self.prompts.append((prompt, system))
        return self.reply


class TestWithTheModel:
    def use(self, monkeypatch, backend):
        monkeypatch.setattr(assistant, "_backend", lambda use_model: backend if use_model else None)
        return backend

    def test_clear_questions_never_reach_the_model(self, monkeypatch):
        backend = self.use(monkeypatch, FakeBackend("bond_price"))
        answer = ai.ask("What is the monthly repayment on a loan of 50000 over 10 years at 8%?")
        assert answer.solver == "loan_repayment" and not answer.used_model
        assert backend.calls == [] and backend.prompts == []

    def test_the_model_chooses_when_the_wording_does_not(self, monkeypatch):
        backend = self.use(monkeypatch, FakeBackend("future_value"))
        answer = ai.ask("I have 5000 and leave it for 12 years at 9%. What do I end up with?")
        assert answer.solved and answer.solver == "future_value" and answer.used_model
        assert answer.value == pytest.approx(5000 * 1.09 ** 12)
        assert "chosen by the language model" in " ".join(answer.notes)
        assert len(backend.calls) == 1 and len(backend.calls[0]) == len(SOLVERS)
        assert not ai.ask("I have 5000 and leave it for 12 years at 9%.", use_model=False).solved

    def test_questions_in_words_go_to_the_model_as_an_actuary(self, monkeypatch):
        backend = self.use(monkeypatch, FakeBackend(reply="The PAA is a simplified model."))
        answer = ai.ask("What does ACTUNEO do for IFRS 17?")
        assert answer.kind == "explanation" and answer.used_model and not answer.solved
        assert answer.text == "The PAA is a simplified model." and answer.sources
        prompt, system = backend.prompts[0]
        assert system == ACTUARY_PROMPT and "qualified actuary" in system
        assert "Never invent figures" in system and "Do not do arithmetic" in system
        assert prompt.startswith("Passages from ACTUNEO's documentation:\n[1] ")
        assert answer.sources[0].text in prompt
        assert prompt.endswith("Question: What does ACTUNEO do for IFRS 17?")
        text = answer.explain()
        assert "written by the language model, not calculated by ACTUNEO" in text
        assert "Passages of ACTUNEO's documentation it drew on" in text
        assert "*Written by the language model" in answer.to_markdown()
        assert backend.calls == []          # no numbers, so no calculation was looked for

    def test_general_questions_use_the_models_own_knowledge(self, monkeypatch):
        backend = self.use(monkeypatch, FakeBackend(reply="An actuary measures risk."))
        answer = ai.ask("Xyzzy plugh quux?")
        assert answer.kind == "explanation" and answer.sources == []
        assert "answer from your own actuarial knowledge" in backend.prompts[0][0]
        part = ai.ask("(a) Present value of 1000 due in 5 years at 10%? (b) Why discount at all?")
        assert [p.kind for p in part.parts] == ["solution", "explanation"]

    def test_a_number_from_the_model_must_be_in_the_question(self, monkeypatch):
        question = "Loan repayment: I borrowed 9000, the rate is 10%, it runs a decade (10)."
        self.use(monkeypatch, FakeBackend("loan_repayment", {"years": 10}))
        accepted = ai.ask(question)
        assert accepted.solved and accepted.arguments["years"] == 10 and accepted.used_model
        assert "picked out by the language model" in " ".join(accepted.notes)
        self.use(monkeypatch, FakeBackend("loan_repayment", {"years": 25}))
        refused = ai.ask(question)
        assert refused.kind == "need_more" and "the term of the loan" in refused.text

    def test_a_failing_model_falls_back(self, monkeypatch):
        class Broken:
            def call_tools(self, *args, **kwargs):
                raise ai.BackendError("model file not found")

            def generate(self, *args, **kwargs):
                raise ai.BackendError("model file not found")
        self.use(monkeypatch, Broken())
        answer = ai.ask("Tell me about 5000 over 12 years")
        assert not answer.solved and "could not be used" in " ".join(answer.notes)
        words = ai.ask("What does ACTUNEO do for IFRS 17?")
        assert words.kind == "reference" and "could not be used" in " ".join(words.notes)


class TestCommandLine:
    def run(self, *arguments):
        from actuneo.ai.cli import main
        return main(list(arguments))

    def test_ask(self, capsys):
        assert self.run("ask", "What", "is", "the", "present", "value", "of", "1000", "due",
                        "in", "5", "years", "at", "10%?", "--no-model") == 0
        printed = capsys.readouterr().out
        assert "Present value of a single payment" in printed and "620.92" in printed

    def test_ask_with_a_data_file(self, tmp_path, capsys):
        path = tmp_path / "flows.csv"
        pd.DataFrame({"flow": [-1000, 300, 400, 500]}).to_csv(path, index=False)
        assert self.run("ask", "NPV at 8%?", "--data", str(path), "--no-model") == 0
        assert "NPV = 17.63" in capsys.readouterr().out

    def test_topics(self, capsys):
        assert self.run("topics") == 0
        printed = capsys.readouterr().out
        assert "Chain ladder reserve" in printed and printed.count("e.g.") == len(SOLVERS)

    def test_chat(self, monkeypatch, capsys):
        typed = iter(["", "topics", "Convert 1,000,000 ZWL to ZiG", "quit"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(typed))
        assert self.run("chat", "--no-model") == 0
        printed = capsys.readouterr().out
        assert "ZiG 400.20" in printed and "Expectation of life" in printed

        def closed(prompt):
            raise EOFError
        monkeypatch.setattr("builtins.input", closed)
        assert self.run("chat") == 0
