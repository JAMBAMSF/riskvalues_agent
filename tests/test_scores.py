import importlib

def test_build_recommendations_order_and_details(monkeypatch):
    planner = importlib.import_module("agent.planner")
    models = importlib.import_module("agent.models")

    # Deterministic fixtures
    def fake_overview(t):
        return {
            "Name": {"MSFT": "Microsoft Corporation", "AAPL": "Apple Inc."}.get(t, f"{t} Corp"),
            "Sector": "Information Technology",
            "Beta": 0.9 if t == "MSFT" else 1.1,
            "ProfitMargin": 0.28 if t == "MSFT" else 0.22,
            "MarketCapitalization": 3_000_000_000_000 if t == "MSFT" else 2_500_000_000_000,
        }

    def fake_osi(t):
        return {
            "carbon_emissions": 10.0 if t == "MSFT" else 15.0,
            "deforestation": 0.1 if t == "MSFT" else 0.2,
            "board_diversity": 35.0 if t == "MSFT" else 28.0,
            "female_board_pct": 32.0 if t == "MSFT" else 25.0,
        }

    # Patch planner tool calls
    monkeypatch.setattr(planner, "fetch_company_overview", fake_overview)
    monkeypatch.setattr(planner, "fetch_sustainability", fake_osi)

    # Patch LLM used by draft_email_from_recs (not used here but safe)
    monkeypatch.setattr(planner, "get_llm", lambda: (lambda s: "ok"))

    # Tiny universe
    universe = [
        {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Information Technology"},
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Information Technology"},
    ]

    recs = planner.build_recommendations(universe, risk="low", prefs=["climate", "diversity"], top_k=2, explain=True)
    assert len(recs) == 2

    # Expect MSFT ahead of AAPL given fixtures
    assert recs[0]["ticker"] == "MSFT"
    assert recs[1]["ticker"] == "AAPL"

    # Composite math is weighted avg of risk/values (0.5/0.5)
    m = recs[0]
    comp_expected = 0.5 * m["risk_score"] + 0.5 * m["values_score"]
    assert abs(m["composite"] - comp_expected) < 1e-9

    # Detail fields present with raw inputs & mapping used
    d = m.get("detail", {})
    assert "risk" in d and "values" in d
    assert set(d["risk"].keys()) == {"beta", "profit_margin", "market_cap"}
    vals = d["values"]
    assert "climate" in vals and "diversity" in vals
    assert vals["climate"]["field"] in ("carbon_emissions", "emissions", "ghg_emissions", "ghg")
    assert isinstance(vals["diversity"]["value"], (int, float))