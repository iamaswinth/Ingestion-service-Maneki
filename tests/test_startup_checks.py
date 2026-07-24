"""app/startup_checks.py — refusing to boot misconfigured.

The two failures worth catching here are asymmetric: an unset
INTERNAL_SERVICE_TOKEN fails *loudly* (every request 401s), while an unset
ANTHROPIC_API_KEY with question generation enabled fails *silently* — ingests
keep succeeding, just without the questions that make retrieval work.
"""

import pytest

from app import config as config_module
from app.startup_checks import ConfigurationError, validate_settings


@pytest.fixture
def configured(monkeypatch):
    """A fully-valid production config; individual tests break one thing."""

    def _apply(**overrides):
        values = {
            "environment": "production",
            "internal_service_token": "internal",
            "database_url": "postgresql://db.example.com/prod",
            "question_gen_enabled": True,
            "sales_script_enabled": True,
            "anthropic_api_key": "sk-ant-x",
        }
        values.update(overrides)
        for name, value in values.items():
            monkeypatch.setattr(config_module.settings, name, value)

    return _apply


def test_a_fully_configured_production_boots(configured):
    configured()
    validate_settings()


def test_unset_internal_token_is_fatal(configured):
    configured(internal_service_token="")
    with pytest.raises(ConfigurationError, match="INTERNAL_SERVICE_TOKEN"):
        validate_settings()


def test_localhost_database_url_is_fatal(configured):
    configured(database_url="postgresql://postgres:postgres@localhost:5433/postgres")
    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        validate_settings()


class TestConditionalAnthropicKey:
    """The key is only required by the features that use it — the bug being
    caught is enabling one and expecting it to work without a key."""

    def test_question_generation_enabled_without_a_key_is_fatal(self, configured):
        configured(anthropic_api_key="", question_gen_enabled=True, sales_script_enabled=False)
        with pytest.raises(ConfigurationError, match="QUESTION_GEN_ENABLED"):
            validate_settings()

    def test_sales_scripts_enabled_without_a_key_is_fatal(self, configured):
        configured(anthropic_api_key="", question_gen_enabled=False, sales_script_enabled=True)
        with pytest.raises(ConfigurationError, match="SALES_SCRIPT_ENABLED"):
            validate_settings()

    def test_both_disabled_needs_no_key(self, configured):
        configured(anthropic_api_key="", question_gen_enabled=False, sales_script_enabled=False)
        validate_settings()


def test_development_only_warns(configured, caplog):
    configured(
        environment="development",
        internal_service_token="",
        anthropic_api_key="",
        database_url="postgresql://postgres:postgres@localhost:5433/postgres",
    )
    validate_settings()
    assert "configuration warning" in caplog.text


def test_staging_is_strict_too(configured):
    configured(environment="staging", internal_service_token="")
    with pytest.raises(ConfigurationError):
        validate_settings()


def test_the_error_names_every_problem_at_once(configured):
    configured(
        internal_service_token="",
        anthropic_api_key="",
        database_url="postgresql://localhost:5433/postgres",
    )
    with pytest.raises(ConfigurationError) as exc:
        validate_settings()

    message = str(exc.value)
    assert "INTERNAL_SERVICE_TOKEN" in message
    assert "DATABASE_URL" in message
    assert "ANTHROPIC_API_KEY" in message
