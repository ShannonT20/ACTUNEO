"""
Build ACTUNEO's file of UN life tables for African countries.

Reads the single-age (complete) life tables of the UN World Population
Prospects 2024, medium variant, for males and females, keeps the African
countries for one year, and writes
``actuneo/mortality/data/un_wpp_2024/africa.csv``.

Download the two source files first (about 200 MB each) from
https://population.un.org/wpp/downloads, CSV format, "Life tables":

    WPP2024_Life_Table_Complete_Medium_Male_1950-2023.csv.gz
    WPP2024_Life_Table_Complete_Medium_Female_1950-2023.csv.gz

Then run, from the repository root:

    python tools/build_un_life_tables.py FOLDER_WITH_THE_TWO_FILES [YEAR]

YEAR defaults to 2023, the latest year of estimates in the 2024 revision.
"""

import sys
from pathlib import Path

import pandas as pd

# Geographic subregions of Africa in the WPP location codes
AFRICAN_SUBREGIONS = {
    910: "Eastern Africa",
    911: "Middle Africa",
    912: "Northern Africa",
    913: "Southern Africa",
    914: "Western Africa",
}
COLUMNS = ["LocID", "ISO3_code", "Location", "ParentID", "LocTypeName", "Time", "Sex",
           "AgeGrpStart", "mx", "qx", "lx", "Tx", "ex"]


def build(folder: Path, year: int = 2023) -> Path:
    frames = []
    for sex, code in (("Male", "M"), ("Female", "F")):
        path = folder / f"WPP2024_Life_Table_Complete_Medium_{sex}_1950-2023.csv.gz"
        chunks = pd.read_csv(path, usecols=COLUMNS, chunksize=500_000, low_memory=False)
        for chunk in chunks:
            keep = chunk[(chunk["Time"] == year)
                         & (chunk["ParentID"].isin(AFRICAN_SUBREGIONS))
                         & (chunk["LocTypeName"] == "Country/Area")]
            if len(keep):
                keep = keep.assign(sex=code)
                frames.append(keep)
    data = pd.concat(frames)
    data["region"] = data["ParentID"].map(AFRICAN_SUBREGIONS)
    out = data.rename(columns={"ISO3_code": "iso3", "Location": "country", "Time": "year",
                               "AgeGrpStart": "age"})
    out = out[["iso3", "country", "region", "year", "sex", "age", "mx", "qx", "lx", "Tx", "ex"]]
    out = out.sort_values(["country", "sex", "age"]).reset_index(drop=True)
    target = Path(__file__).resolve().parents[1] / "actuneo" / "mortality" / "data" / \
        "un_wpp_2024" / "africa.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False, float_format="%.8g")
    return target


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2023
    print(build(folder, year))
