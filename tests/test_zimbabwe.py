"""
Tests for the zimbabwe module.
"""

import pytest
from actuneo import zimbabwe
from actuneo.mortality import MortalityTable


def test_catalogue_lists_every_shipped_table():
    catalogue = zimbabwe.mortality_tables()
    assert sorted(catalogue.index) == MortalityTable.zimbabwe_2023_tables()
    assert set(catalogue["sex"]) == {"male", "female", "combined"}
    assert catalogue.loc["male_assured_lives", "min_age"] == 20
    assert catalogue.loc["male_assured_lives", "max_age"] == 100
    assert catalogue.loc["group_life_male", "max_age"] == 70
    assert catalogue.loc["post_retirement_pensions_female", "min_age"] == 76


def test_load_mortality_table():
    table = zimbabwe.load_mortality_table("funeral_spouses")
    assert table.metadata["country"] == "Zimbabwe"
    assert table.qx(46) == pytest.approx(0.0033338)


def test_currency_conversion():
    assert zimbabwe.CURRENCY_CODE == "ZWG"
    assert zimbabwe.zwl_to_zwg(2498.7242) == pytest.approx(1.0)
    assert zimbabwe.zwl_to_zwg(1_000_000) == pytest.approx(400.2043)
