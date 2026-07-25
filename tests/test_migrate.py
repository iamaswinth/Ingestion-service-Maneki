"""app/migrate.py — the schema migration runner.

Runs against the real Postgres the rest of the suite uses, in a throwaway
schema per test, because the properties worth testing (exactly-once, atomicity,
advisory locking) are properties of the database, not of Python.
"""

import asyncio
import uuid

import asyncpg
import pytest
from urllib.parse import quote

from app import config as config_module
from app.migrate import MigrationError, apply_pending, discover, status


@pytest.fixture
async def scratch_schema(monkeypatch):
    """An empty Postgres schema, dropped afterwards — so each test sees a
    database with no migrations applied."""
    name = f"migrate_test_{uuid.uuid4().hex[:12]}"

    conn = await asyncpg.connect(config_module.settings.database_url)
    await conn.execute(f"CREATE SCHEMA {name}")
    await conn.close()

    global _active_schema
    _active_schema = name
    conn = await asyncpg.connect(_scratch_dsn(name))
    try:
        yield conn
    finally:
        await conn.close()
        cleanup = await asyncpg.connect(config_module.settings.database_url)
        await cleanup.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE")
        await cleanup.close()


@pytest.fixture
def migration_dir(tmp_path):
    def _write(**files: str):
        for name, sql in files.items():
            (tmp_path / f"{name}.sql").write_text(sql, encoding="utf-8")
        return tmp_path

    return _write


# asyncpg.Connection uses __slots__, so the active schema is tracked here
# rather than attached to the connection object.
_active_schema: str = ""


def _scratch_dsn(schema: str) -> str:
    # `public` stays on the path: CREATE EXTENSION installs pgvector into
    # whichever schema already holds it (public, in practice, because app/db.py
    # bootstraps it there), so the `vector` type is only resolvable if public is
    # visible. Without it the baseline fails with 'type "vector" does not
    # exist' — but only when another test created the extension first, which is
    # why this passes in isolation and fails in the full suite.
    url = config_module.settings.database_url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}options={quote(f'-c search_path={schema},public', safe='')}"


async def _tables(conn) -> set[str]:
    rows = await conn.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname = $1", _active_schema
    )
    return {row["tablename"] for row in rows}


class TestDiscovery:
    def test_orders_by_zero_padded_filename(self, migration_dir):
        # The bug zero-padding exists to prevent: "10" sorts before "2".
        directory = migration_dir(
            **{"0002_b": "SELECT 1;", "0010_c": "SELECT 1;", "0001_a": "SELECT 1;"}
        )
        assert [m.version for m in discover(directory)] == ["0001_a", "0002_b", "0010_c"]

    def test_missing_directory_is_an_error(self, tmp_path):
        with pytest.raises(MigrationError, match="not found"):
            discover(tmp_path / "nope")

    def test_checksum_ignores_line_ending_differences(self, tmp_path):
        # A Windows checkout and a Linux CI runner must agree, or every
        # migration looks edited-after-apply on one of them.
        (tmp_path / "0001_a.sql").write_text("SELECT 1;\nSELECT 2;\n", encoding="utf-8")
        unix = discover(tmp_path)[0].checksum
        (tmp_path / "0001_a.sql").write_bytes(b"SELECT 1;\r\nSELECT 2;\r\n")
        assert discover(tmp_path)[0].checksum == unix


class TestApply:
    async def test_applies_pending_migrations_in_order(self, scratch_schema, migration_dir):
        directory = migration_dir(
            **{
                "0001_a": "CREATE TABLE alpha (id INT);",
                "0002_b": "CREATE TABLE beta (id INT);",
            }
        )
        applied = await apply_pending(scratch_schema, directory)

        assert applied == ["0001_a", "0002_b"]
        assert {"alpha", "beta"} <= await _tables(scratch_schema)

    async def test_second_run_applies_nothing(self, scratch_schema, migration_dir):
        directory = migration_dir(**{"0001_a": "CREATE TABLE alpha (id INT);"})
        await apply_pending(scratch_schema, directory)

        # Not merely "doesn't error" — the CREATE TABLE here has no IF NOT
        # EXISTS, so re-running it would raise. Exactly-once is what's tested.
        assert await apply_pending(scratch_schema, directory) == []

    async def test_only_the_new_migration_runs_on_a_later_deploy(
        self, scratch_schema, migration_dir
    ):
        directory = migration_dir(**{"0001_a": "CREATE TABLE alpha (id INT);"})
        await apply_pending(scratch_schema, directory)

        (directory / "0002_b.sql").write_text("CREATE TABLE beta (id INT);", encoding="utf-8")
        assert await apply_pending(scratch_schema, directory) == ["0002_b"]

    async def test_a_failing_migration_is_not_recorded_or_half_applied(
        self, scratch_schema, migration_dir
    ):
        directory = migration_dir(
            **{
                "0001_a": "CREATE TABLE alpha (id INT);",
                "0002_bad": "CREATE TABLE beta (id INT); SELECT this_is_not_valid();",
            }
        )
        with pytest.raises(asyncpg.PostgresError):
            await apply_pending(scratch_schema, directory)

        tables = await _tables(scratch_schema)
        assert "alpha" in tables  # the migration that succeeded stands
        assert "beta" not in tables  # the failing one rolled back entirely

        rows = await scratch_schema.fetch("SELECT version FROM schema_migrations")
        assert [r["version"] for r in rows] == ["0001_a"]

    async def test_a_fixed_migration_can_then_be_applied(self, scratch_schema, migration_dir):
        # Recovering forward after the failure above must actually work.
        directory = migration_dir(**{"0001_bad": "SELECT this_is_not_valid();"})
        with pytest.raises(asyncpg.PostgresError):
            await apply_pending(scratch_schema, directory)

        (directory / "0001_bad.sql").write_text("CREATE TABLE alpha (id INT);", encoding="utf-8")
        assert await apply_pending(scratch_schema, directory) == ["0001_bad"]


class TestImmutableHistory:
    async def test_editing_an_applied_migration_is_refused(self, scratch_schema, migration_dir):
        directory = migration_dir(**{"0001_a": "CREATE TABLE alpha (id INT);"})
        await apply_pending(scratch_schema, directory)

        (directory / "0001_a.sql").write_text("CREATE TABLE gamma (id INT);", encoding="utf-8")
        with pytest.raises(MigrationError, match="immutable"):
            await apply_pending(scratch_schema, directory)

    async def test_an_unapplied_migration_may_still_be_edited(self, scratch_schema, migration_dir):
        # Only *applied* history is frozen — editing one you haven't shipped
        # yet is normal work.
        directory = migration_dir(**{"0001_a": "CREATE TABLE alpha (id INT);"})
        await apply_pending(scratch_schema, directory)

        (directory / "0002_b.sql").write_text("CREATE TABLE wrong (id INT);", encoding="utf-8")
        (directory / "0002_b.sql").write_text("CREATE TABLE beta (id INT);", encoding="utf-8")
        assert await apply_pending(scratch_schema, directory) == ["0002_b"]


class TestConcurrency:
    async def test_two_simultaneous_runners_do_not_double_apply(
        self, scratch_schema, migration_dir
    ):
        # Two replicas deploying at once. Without the advisory lock both read
        # an empty schema_migrations and both run the CREATE TABLE.
        directory = migration_dir(**{"0001_a": "CREATE TABLE alpha (id INT);"})

        second = await asyncpg.connect(_scratch_dsn(_active_schema))
        try:
            results = await asyncio.gather(
                apply_pending(scratch_schema, directory),
                apply_pending(second, directory),
            )
        finally:
            await second.close()

        # Exactly one runner did the work; the other found it already done.
        assert sorted(len(r) for r in results) == [0, 1]
        rows = await scratch_schema.fetch("SELECT version FROM schema_migrations")
        assert len(rows) == 1


class TestStatus:
    async def test_reports_applied_and_pending(self, scratch_schema, migration_dir):
        directory = migration_dir(**{"0001_a": "CREATE TABLE alpha (id INT);"})
        await apply_pending(scratch_schema, directory)

        (directory / "0002_b.sql").write_text("CREATE TABLE beta (id INT);", encoding="utf-8")
        assert await status(scratch_schema, directory) == [("0001_a", True), ("0002_b", False)]

    async def test_status_does_not_create_anything(self, scratch_schema, migration_dir):
        directory = migration_dir(**{"0001_a": "CREATE TABLE alpha (id INT);"})

        assert await status(scratch_schema, directory) == [("0001_a", False)]
        # Reporting status must leave the database exactly as it found it.
        assert await _tables(scratch_schema) == set()


class TestRealBaseline:
    async def test_the_checked_in_baseline_applies_cleanly(self, scratch_schema):
        # Guards the actual shipped migrations, not just synthetic fixtures.
        assert await apply_pending(scratch_schema) == ["0001_baseline", "0002_chunk_site_index"]
        assert {"jobs", "pages", "chunks", "sales_scripts"} <= await _tables(scratch_schema)

    async def test_the_baseline_is_safe_on_a_database_that_already_has_the_tables(
        self, scratch_schema
    ):
        # The real situation when migrations landed: every existing database
        # already had the full schema (applied by the old inline DDL) but no
        # schema_migrations record. Simulated by applying the baseline, then
        # forgetting it was applied.
        await apply_pending(scratch_schema)
        await scratch_schema.execute("DROP TABLE schema_migrations")

        assert await apply_pending(scratch_schema) == ["0001_baseline", "0002_chunk_site_index"]
