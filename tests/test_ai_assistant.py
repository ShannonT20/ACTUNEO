"""
Tests of actuneo.ai.ask: reading a question, choosing the calculation, and the
worked solution.

Every solver's answer is compared with a direct call to the library. The
language model is replaced by a stand-in.
"""

import importlib
import warnings

import pytest

from actuneo import ai, zimbabwe
from actuneo.ai.assistant import extract_arguments, normalise, rank_solvers
from actuneo.ai.benchmark import arguments_match
from actuneo.ai.solvers import GLOSSARY, SOLVERS
from actuneo.finance import Bond, Loan
from actuneo.life import LifeAssurance
from actuneo.loss_reserving import ChainLadder, MackChainLadder, load_genins, load_raa
from actuneo.mortality import MortalityTable
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
        both = ai.ask("Chain ladder reserve for RAA and GenIns")
        assert both.arguments == {"triangle": "RAA"} and "More than one" in " ".join(both.notes)

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
        assert off_table.kind == "need_more" and "covers ages 15 to 104" in off_table.text
        zim = ai.ask("Value a pension of 6000 a year from age 65 at 5% on the Zimbabwe tables")
        assert zim.kind == "need_more" and "76" in zim.text

    def test_unrelated_questions(self):
        for question in ["Good morning, can you help me?", "", "What is the weather?",
                         "What is the probability that it rains in Bulawayo tomorrow?"]:
            answer = ai.ask(question)
            assert answer.kind == "unknown" and "I can work out:" in answer.text
            assert "Chain ladder reserve" in answer.text

    def test_definitions_come_from_the_glossary(self):
        ibnr = ai.ask("Explain what IBNR means.")
        assert ibnr.kind == "definition" and ibnr.text == GLOSSARY["ibnr"]
        assert "incurred but not reported" in ibnr.text
        better = ai.ask("Which is better, chain ladder or Bornhuetter-Ferguson?")
        assert better.kind == "definition"
        assert GLOSSARY["chain ladder"] in better.text and GLOSSARY["bornhuetter"] in better.text
        assert "does not choose between them" in better.text
        due = ai.ask("What is an annuity-due?")
        assert due.text.startswith("An annuity-due is")
        # A calculation that can be done is done, even if it reads like a definition
        assert ai.ask("What is the Mack standard error for the GenIns triangle?").solved

    def test_ranking(self):
        ranked = rank_solvers("how uncertain is the chain ladder reserve for raa?")
        assert ranked[0][1].name == "mack_standard_error" and ranked[1][1].name == "chain_ladder_reserve"
        assert rank_solvers("hello there") == []
        assert len(ai.topics()) == len(SOLVERS) == 14


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

    def test_mortality(self):
        females = MortalityTable.from_south_africa("saifl98")
        males = MortalityTable.from_south_africa("saiml98")
        assured = MortalityTable.from_south_africa("sa8590")
        assert self.value("Life expectancy of a 45 year old woman") == \
            pytest.approx(females.life_expectancy(45))
        assert self.value("Life expectancy of a 30 year old man") == \
            pytest.approx(assured.life_expectancy(30))
        assert self.value("Probability that a 30 year old survives 20 years") == \
            pytest.approx(assured.npx(30, 20))
        assert self.value("Value a pension of 6000 a year from age 65 at 5% interest.") == \
            pytest.approx(6000 * pension_annuity(males, 65, 0.05))
        assert self.value("Premium for a 20-year term assurance of 100000 for a 35 year old at 6%") == \
            pytest.approx(100000 * LifeAssurance(assured, interest_rate=0.06)
                          .net_annual_premium(35, 20, "term"))

    def test_zimbabwe_basis(self):
        table = MortalityTable.from_zimbabwe_2023("male_assured_lives")
        answer = ai.ask("Life expectancy of a 40 year old man on the Zimbabwe tables")
        assert answer.arguments["basis"] == "zimbabwe"
        assert answer.value == pytest.approx(table.life_expectancy(40))
        assert "Zimbabwe 2023 male assured lives" in answer.explain()
        woman = ai.ask("Life expectancy of a 40 year old woman in Zimbabwe")
        assert "no female assured-lives table" in woman.explain()
        assert self.value("Chance a 30 year old survives 10 years, Zimbabwe basis") == \
            pytest.approx(table.npx(30, 10))

    def test_reserving(self):
        assert self.value("Chain ladder IBNR for the RAA triangle") == \
            pytest.approx(ChainLadder(load_raa()).total_ibnr)
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
        assert "out of 100,000 at age 15" in life and "n_p_x = l_(x+n) / l_x" in life


class FakeBackend:
    """A stand-in for the language model that returns a scripted choice."""

    def __init__(self, name, arguments=None):
        self.name, self.arguments, self.calls = name, arguments or {}, []

    def call_tools(self, prompt, tools, system=None, allow_none=True):
        self.calls.append([tool["name"] for tool in tools])
        return ai.ToolCall(self.name, dict(self.arguments))


class TestWithTheModel:
    def use(self, monkeypatch, backend):
        monkeypatch.setattr(assistant, "_backend", lambda use_model: backend if use_model else None)
        return backend

    def test_clear_questions_never_reach_the_model(self, monkeypatch):
        backend = self.use(monkeypatch, FakeBackend("bond_price"))
        answer = ai.ask("What is the monthly repayment on a loan of 50000 over 10 years at 8%?")
        assert answer.solver == "loan_repayment" and backend.calls == [] and not answer.used_model

    def test_the_model_chooses_when_the_wording_does_not(self, monkeypatch):
        backend = self.use(monkeypatch, FakeBackend("future_value"))
        answer = ai.ask("I have 5000 and leave it for 12 years at 9%. What do I end up with?")
        assert answer.solved and answer.solver == "future_value" and answer.used_model
        assert answer.value == pytest.approx(5000 * 1.09 ** 12)
        assert "chosen by the language model" in " ".join(answer.notes)
        assert len(backend.calls) == 1 and len(backend.calls[0]) == len(SOLVERS)
        assert not ai.ask("I have 5000 and leave it for 12 years at 9%.", use_model=False).solved

    def test_the_model_saying_none_is_respected(self, monkeypatch):
        self.use(monkeypatch, FakeBackend(None))
        assert ai.ask("What is the probability that it rains tomorrow?").kind == "unknown"

    def test_a_number_from_the_model_must_be_in_the_question(self, monkeypatch):
        question = "Loan repayment: I borrowed 9000, the rate is 10%, it runs a decade (10)."
        self.use(monkeypatch, FakeBackend("loan_repayment", {"years": 10}))
        accepted = ai.ask(question)
        assert accepted.solved and accepted.arguments["years"] == 10 and accepted.used_model
        assert "picked out by the language model" in " ".join(accepted.notes)
        self.use(monkeypatch, FakeBackend("loan_repayment", {"years": 25}))
        refused = ai.ask(question)
        assert refused.kind == "need_more" and "the term of the loan" in refused.text

    def test_a_failing_model_falls_back_to_the_rules(self, monkeypatch):
        class Broken:
            def call_tools(self, *args, **kwargs):
                raise ai.BackendError("model file not found")
        self.use(monkeypatch, Broken())
        answer = ai.ask("Tell me about 5000 over 12 years")
        assert answer.kind == "unknown" and "could not be used" in " ".join(answer.notes)


class TestCommandLine:
    def run(self, *arguments):
        from actuneo.ai.cli import main
        return main(list(arguments))

    def test_ask(self, capsys):
        assert self.run("ask", "What", "is", "the", "present", "value", "of", "1000", "due",
                        "in", "5", "years", "at", "10%?", "--no-model") == 0
        printed = capsys.readouterr().out
        assert "Present value of a single payment" in printed and "620.92" in printed

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
