import importlib
import io
import contextlib

def _patch_llm(monkeypatch):
    # Return a stable "email draft" string without requiring any real LLM libs/keys
    def dummy_llm():
        return lambda prompt: "EMAIL DRAFT (test): " + prompt.splitlines()[0][:80]
    planner = importlib.import_module("agent.planner")
    monkeypatch.setattr(planner, "get_llm", dummy_llm)

def _patch_tools_for_cli(monkeypatch):
    tools = importlib.import_module("agent.tools")

    def fake_overview(t):
        return {
            "Name": {"MSFT": "Microsoft Corporation", "AAPL": "Apple Inc."}.get(t, f"{t} Corp"),
            "Sector": "Information Technology",
            "Beta": 0.9 if t == "MSFT" else 1.1,
            "ProfitMargin": 0.28 if t == "MSFT" else 0.22,
            "MarketCapitalization": 3_000_000_000_000 if t == "MSFT" else 2_500_000_000_000,
            "Description": "Our mission is to empower every person and every organization on the planet to achieve more.",
        }

    def fake_osi(t):
        return {
            "carbon_emissions": 10.0 if t == "MSFT" else 15.0,
            "deforestation": 0.1 if t == "MSFT" else 0.2,
            "board_diversity": 35.0 if t == "MSFT" else 28.0,
            "female_board_pct": 32.0 if t == "MSFT" else 25.0,
            "founded": "1975" if t == "MSFT" else "1976",
        }

    def fake_sec(t, limit=1):
        return [{"title": f"{t} 10-K (Test)", "url": f"https://sec.test/{t}/10-K"}] if limit else []

    # Patch tool funcs used by app.answer_question and planner.build_recommendations
    monkeypatch.setattr(tools, "fetch_company_overview", fake_overview)
    monkeypatch.setattr(tools, "fetch_sustainability", fake_osi)
    monkeypatch.setattr(tools, "search_sec_filings", fake_sec)
    monkeypatch.setattr(tools, "ensure_sp500_universe", lambda: [
        {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Information Technology"},
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Information Technology"},
    ])

def test_cmd_recommend_prints_sorted_table_and_email(monkeypatch, capsys):
    _patch_llm(monkeypatch)
    _patch_tools_for_cli(monkeypatch)

    app = importlib.import_module("app")
    # capture stdout
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        app.cmd_recommend(risk="low", values="climate,diversity", k=2, out=None, explain=True)
    out = buf.getvalue()

    assert "rank\tticker\tname\trisk_score\tvalues_score\tcomposite" in out
    # two rows, sorted, and details printed
    assert "\n1\tMSFT" in out and "\n2\tAAPL" in out
    assert "Details (top 3):" in out
    assert "Email draft" in out

def test_answer_question_mission_and_snapshot(monkeypatch):
    _patch_llm(monkeypatch)
    _patch_tools_for_cli(monkeypatch)

    app = importlib.import_module("app")
    out = app.answer_question("What's the mission and sustainability for MSFT?")
    assert "## MSFT Overview" in out
    assert "Mission:" in out
    assert "Sustainability snapshot:" in out
    # We include the 10-K link when available
    assert "10-K (Test)" in out