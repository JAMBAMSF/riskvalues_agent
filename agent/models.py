# agent/models.py
# -----------------------------------------------------------------------------
# These are my typed models for normalizing tool responses
# (Alpha Vantage financials and OSi sustainability-style data).
#
# What it does:
#   - Defines Pydantic models for the fields I actually use in the agent.
#   - Coerces messy API values (strings like "NaN", "None", or numbers
#     with commas) into clean float|None.
#   - Ignores all extra keys from the upstream APIs to keep things tight.
#
# Why I wrote it this way:
#   - Free-tier APIs are noisy and inconsistent.
#   - I wanted type-checked, predictable data to plug into scoring.
#   - Using Pydantic validators saves me from writing the same
#     "string to float or None" logic everywhere else in the code.
#
# Important:
#   - This is just a thin schema layer, not a full data model.
#   - Only the fields relevant to risk (beta, margins, market cap)
#     and values (climate, deforestation, diversity) are included.
#   - Everything else from the APIs is ignored on purpose.
#
# Author: Jaxon Archer <Jaxon.Archer.MBA.MSF@gmail.com>
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, field_validator


def _to_float_or_none(v: Any) -> Optional[float]:
    """I coerce typical API values ('1.23', 1.23, 'None', '', None) to float|None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() == "none" or s.lower() == "null" or s == "NaN":
            return None
        # remove thousands separators if they ever appear
        s = s.replace(",", "")
        try:
            return float(s)
        except Exception:
            return None
    return None


class AlphaVantageOverview(BaseModel):
    """Subset of Alpha Vantage OVERVIEW fields I actually use.
    I ignore extra keys from the API by design.
    """
    model_config = ConfigDict(extra="ignore")

    Name: Optional[str] = None
    Sector: Optional[str] = None
    Beta: Optional[float] = None
    ProfitMargin: Optional[float] = None
    MarketCapitalization: Optional[float] = None
    Description: Optional[str] = None

    @field_validator("Beta", "ProfitMargin", "MarketCapitalization", mode="before")
    @classmethod
    def _coerce_numeric(cls, v: Any) -> Optional[float]:
        return _to_float_or_none(v)


class OsiCompany(BaseModel):
    """OSi-like sustainability fields I care about.
    Lower = better for climate/deforestation; higher = better for diversity.
    """
    model_config = ConfigDict(extra="ignore")

    carbon_emissions: Optional[float] = None
    emissions: Optional[float] = None
    ghg_emissions: Optional[float] = None
    ghg: Optional[float] = None
    deforestation: Optional[float] = None
    forest_policy: Optional[float] = None
    supply_chain_deforestation: Optional[float] = None
    board_diversity: Optional[float] = None
    diversity: Optional[float] = None
    gender_diversity: Optional[float] = None
    female_board_pct: Optional[float] = None

    @field_validator(
        "carbon_emissions",
        "emissions",
        "ghg_emissions",
        "ghg",
        "deforestation",
        "forest_policy",
        "supply_chain_deforestation",
        "board_diversity",
        "diversity",
        "gender_diversity",
        "female_board_pct",
        mode="before",
    )
    @classmethod
    def _coerce_all_numeric(cls, v: Any) -> Optional[float]:
        return _to_float_or_none(v)