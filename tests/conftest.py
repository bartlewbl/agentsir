import pytest


@pytest.fixture(autouse=True)
def no_lambda_functions(monkeypatch):
    """Keep a developer's .env from sending the offline tests to AWS."""
    monkeypatch.delenv("LAMBDA_TOOL_FUNCTIONS", raising=False)
