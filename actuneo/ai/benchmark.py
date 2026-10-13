"""
Benchmark of tool calling

A fixed set of actuarial requests, each with the function that should be
chosen and the arguments it should be given. :func:`benchmark` runs them
through a backend and reports how often the function and the arguments are
right, how often an unrelated request is correctly left alone, and how long
each takes. Use it to compare models on your own machine before choosing one.

The functions here are descriptions only: nothing is calculated. The set is
small and was written for this purpose, so a score is a guide to a model's
reliability at this task, not a general measure of the model.
"""

import time
from typing import Any, Dict, List, Optional

import pandas as pd

from .backends import Backend

_RATE = {"type": "number", "description": "as a decimal, for example 0.05 for 5%"}


def _tool(name: str, description: str, **parameters) -> Dict[str, Any]:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": parameters,
                           "required": [key for key in parameters if key != "sex"]}}


#: Twelve functions, more than a model is shown at once, so that the
#: shortlisting of tools is exercised as well.
TOOLS: List[Dict[str, Any]] = [
    _tool("life_expectancy", "Expectation of life, how long a person is expected to live, at an age",
          age={"type": "integer"}, sex={"type": "string", "enum": ["M", "F"]}),
    _tool("survival_probability", "Probability that a person of a given age survives, or "
          "lives, a number of years", age={"type": "integer"}, years={"type": "integer"}),
    _tool("annuity_value", "Value or cost of an annuity or pension of an annual amount paid "
          "for life from an age", age={"type": "integer"}, annual_amount={"type": "number"},
          interest_rate=_RATE),
    _tool("term_assurance_premium", "Premium for a term assurance (life cover) of a sum assured "
          "for a term of years", age={"type": "integer"}, term={"type": "integer"},
          sum_assured={"type": "number"}, interest_rate=_RATE),
    _tool("chain_ladder_reserve", "IBNR claims reserve from a run-off triangle by the "
          "chain ladder method", triangle={"type": "string"}),
    _tool("mack_standard_error", "Mack standard error of the chain ladder reserve for a "
          "triangle", triangle={"type": "string"}),
    _tool("bornhuetter_ferguson_reserve", "Bornhuetter-Ferguson reserve for a triangle from an "
          "expected loss ratio", triangle={"type": "string"}, loss_ratio=_RATE),
    _tool("loan_repayment", "Regular repayment on a loan of an amount over a number of years",
          amount={"type": "number"}, interest_rate=_RATE, years={"type": "integer"}),
    _tool("bond_price", "Price of a bond from its coupon rate, term in years and yield",
          coupon_rate=_RATE, term={"type": "integer"}, yield_rate=_RATE),
    _tool("present_value", "Present value of a single amount due in a number of years",
          amount={"type": "number"}, years={"type": "integer"}, interest_rate=_RATE),
    _tool("pension_liability", "Pension scheme liability for an active member from age, "
          "years of service and salary", age={"type": "integer"}, service={"type": "number"},
          salary={"type": "number"}),
    _tool("zwl_to_zig", "Convert an amount of Zimbabwe dollars (ZWL) to Zimbabwe Gold (ZiG)",
          amount={"type": "number"}),
]

#: (request, expected function or None, expected arguments)
CASES = [
    ("What is the life expectancy of a 40 year old woman?", "life_expectancy",
     {"age": 40, "sex": "F"}),
    ("How long is a man aged 65 expected to live?", "life_expectancy", {"age": 65, "sex": "M"}),
    ("What is the probability that someone aged 30 survives 20 years?", "survival_probability",
     {"age": 30, "years": 20}),
    ("Chance a 50 year old lives another 10 years?", "survival_probability",
     {"age": 50, "years": 10}),
    ("Value a pension of 6000 a year from age 65 at 5% interest.", "annuity_value",
     {"age": 65, "annual_amount": 6000, "interest_rate": 0.05}),
    ("What does an annuity of 12,000 per year for a 60 year old cost at 4% interest?",
     "annuity_value", {"age": 60, "annual_amount": 12000, "interest_rate": 0.04}),
    ("Premium for a 20-year term assurance of 100000 for a 35 year old at 6% interest",
     "term_assurance_premium",
     {"age": 35, "term": 20, "sum_assured": 100000, "interest_rate": 0.06}),
    ("Estimate the IBNR for the RAA triangle with the chain ladder method.",
     "chain_ladder_reserve", {"triangle": "RAA"}),
    ("Run chain ladder on GenIns", "chain_ladder_reserve", {"triangle": "GenIns"}),
    ("What is the Mack standard error of the reserve for the GenIns triangle?",
     "mack_standard_error", {"triangle": "GenIns"}),
    ("Bornhuetter-Ferguson reserve for the RAA triangle with a loss ratio of 75%",
     "bornhuetter_ferguson_reserve", {"triangle": "RAA", "loss_ratio": 0.75}),
    ("Repayment on a loan of 50000 over 10 years at 8% interest?", "loan_repayment",
     {"amount": 50000, "interest_rate": 0.08, "years": 10}),
    ("Price a 10-year bond with a 6% coupon at a yield of 7%.", "bond_price",
     {"coupon_rate": 0.06, "term": 10, "yield_rate": 0.07}),
    ("What is the present value of 1000 due in 5 years at 10% interest?", "present_value",
     {"amount": 1000, "years": 5, "interest_rate": 0.10}),
    ("What is the pension liability for a member aged 45 with 15 years of service earning a "
     "salary of 26400?", "pension_liability", {"age": 45, "service": 15, "salary": 26400}),
    ("Convert 1,000,000 ZWL to ZiG.", "zwl_to_zig", {"amount": 1000000}),
    ("Good morning!", None, {}),
    ("What is the capital of Zimbabwe?", None, {}),
    ("Write a short poem about rain.", None, {}),
    ("Thanks, that's all.", None, {}),
]


def _same(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str):
        return isinstance(actual, str) and expected.lower() == actual.strip().lower()
    try:
        return abs(float(actual) - float(expected)) <= 1e-6 * max(1.0, abs(float(expected)))
    except (TypeError, ValueError):
        return False


def arguments_match(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    """True if every expected argument was given with the expected value."""
    return all(key in actual and _same(value, actual[key]) for key, value in expected.items())


def benchmark(backend: Optional[Backend] = None, verbose: bool = True) -> pd.DataFrame:
    """
    Run the benchmark requests through a backend.

    Args:
        backend: The backend to test (the configured one if omitted)
        verbose: Print each request and the summary as it goes

    Returns:
        One row per request: what was expected, what the model chose,
        whether the function and the arguments were right, and the seconds
        taken. ``attrs["summary"]`` holds the totals:

        - ``function_accuracy``: right function, or correctly none, over all
          requests
        - ``argument_accuracy``: right function *and* arguments, over the
          requests that call for a function
        - ``none_accuracy``: unrelated requests correctly left alone
        - ``mean_seconds`` per request
    """
    if backend is None:
        from .setup import get_backend
        backend = get_backend()
    if hasattr(backend, "clear_cache"):
        backend.clear_cache()
    rows = []
    for request, expected, expected_arguments in CASES:
        started = time.time()
        error = ""
        try:
            call = backend.call_tools(request, TOOLS)
            name, arguments = call.name, call.arguments
        except Exception as problem:        # a failed call counts as a wrong answer
            name, arguments, error = "<error>", {}, str(problem)[:120]
        seconds = time.time() - started
        function_ok = name == expected
        arguments_ok = function_ok and arguments_match(expected_arguments, arguments)
        rows.append({"request": request, "expected": expected, "chosen": name,
                     "arguments": arguments, "function_ok": function_ok,
                     "arguments_ok": arguments_ok, "seconds": round(seconds, 2),
                     "error": error})
        if verbose:
            mark = "ok " if arguments_ok else ("arg" if function_ok else "NO ")
            print(f"[{mark}] {seconds:5.1f}s  {request[:58]:58s} -> {name} {arguments}")
    table = pd.DataFrame(rows)
    with_function = table[table["expected"].notna()]
    without = table[table["expected"].isna()]
    summary = {
        "requests": len(table),
        "function_accuracy": float(table["function_ok"].mean()),
        "argument_accuracy": float(with_function["arguments_ok"].mean()),
        "none_accuracy": float(without["function_ok"].mean()),
        "mean_seconds": float(table["seconds"].mean()),
    }
    table.attrs["summary"] = summary
    if verbose:
        print(f"Function right: {summary['function_accuracy']:.0%}   "
              f"function and arguments right: {summary['argument_accuracy']:.0%}   "
              f"unrelated requests left alone: {summary['none_accuracy']:.0%}   "
              f"mean {summary['mean_seconds']:.1f}s per request")
    return table
