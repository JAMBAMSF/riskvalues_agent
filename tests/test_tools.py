import importlib
from types import SimpleNamespace

def test_extract_mission_from_description():
    mission_mod = importlib.import_module("agent.mission")
    text = "Acme builds widgets. Our mission is to decarbonize heavy industry while improving safety."
    out = mission_mod.extract_mission_from_description(text)
    assert out.startswith("Our mission is to decarbonize")

def test_search_sec_filings_parses_atom(monkeypatch):
    tools = importlib.import_module("agent.tools")

    atom = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>MSFT 10-K (Test)</title>
        <link rel="alternate" href="https://sec.test/MSFT/10-K"/>
      </entry>
    </feed>"""

    class FakeResp:
        status_code = 200
        text = atom

    # Patch the session's get to return our fake Atom
    monkeypatch.setattr(tools, "S", SimpleNamespace(get=lambda *a, **k: FakeResp()))

    rows = tools.search_sec_filings("MSFT", limit=1)
    assert rows and rows[0]["title"].startswith("MSFT 10-K")
    assert rows[0]["url"].endswith("/MSFT/10-K")