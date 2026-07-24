"""Schema migration runner.

    python -m app.migrate            apply everything pending
    python -m app.migrate --status   show what's applied and what's pending

Deliberately not Alembic. This service has no ORM — the schema is raw SQL over
raw asyncpg — so Alembic's autogenerate has nothing to introspect and every
migration would be a hand-written `op.execute("...")` anyway. That's the
whole of Alembic's value here, in exchange for pulling SQLAlchemy and greenlet
into a service that uses neither.

What it does guarantee:

- **Ordering.** Files in migrations/ are applied in filename order, so the
  zero-padded numeric prefix is the version. Sorting is on the filename as a
  string, which is why the padding matters: `0010` must not sort before `0002`.
- **Exactly once.** Applied versions are recorded in `schema_migrations`.
- **Atomicity per migration.** Each file runs inside its own transaction
  together with its `schema_migrations` insert, so a failure halfway through
  leaves that migration entirely unapplied rather than half-applied and
  unrecorded. Note Postgres allows DDL in a transaction — this is not true of
  every database, and it's what makes the guarantee real here.
- **Safety under concurrent deploys.** A session-level advisory lock means two
  replicas starting at once serialize instead of racing; the loser sees the
  work already done and applies nothing.
- **Immutable history.** Each file's checksum is stored on first apply and
  re-checked afterwards, so editing a migration that has already run is an
  error rather than a silent divergence between environments.

Rolling back is deliberately not supported: a down-migration that has never
been run is not a rollback plan, it's a second untested code path. Recover
forward with a new migration.
"""

import asyncio
import hashlib
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import asyncpg

from .config import settings

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

# Arbitrary but must be stable and distinct from any other advisory lock used
# against this database. The api-gateway uses its own value.
_ADVISORY_LOCK_KEY = 8_472_002

_SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    checksum    TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class Migration(NamedTuple):
    version: str
    path: Path
    sql: str
    checksum: str


class MigrationError(RuntimeError):
    pass


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    if not directory.is_dir():
        raise MigrationError(f"migrations directory not found: {directory}")

    migrations: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        migrations.append(
            Migration(
                version=path.stem,
                path=path,
                sql=sql,
                # Normalize line endings so a Windows checkout and a Linux CI
                # runner agree on the checksum of an unchanged file.
                checksum=hashlib.sha256(sql.replace("\r\n", "\n").encode("utf-8")).hexdigest(),
            )
        )

    versions = [m.version for m in migrations]
    duplicates = {v for v in versions if versions.count(v) > 1}
    if duplicates:
        raise MigrationError(f"duplicate migration versions: {sorted(duplicates)}")
    return migrations


async def _applied(conn: asyncpg.Connection) -> dict[str, str]:
    rows = await conn.fetch("SELECT version, checksum FROM schema_migrations")
    return {row["version"]: row["checksum"] for row in rows}


def _verify_checksums(migrations: list[Migration], applied: dict[str, str]) -> None:
    for migration in migrations:
        recorded = applied.get(migration.version)
        if recorded is not None and recorded != migration.checksum:
            raise MigrationError(
                f"{migration.version} has already been applied but its contents "
                f"have changed since. Migrations are immutable once applied — "
                f"add a new one instead of editing {migration.path.name}."
            )


async def apply_pending(conn: asyncpg.Connection, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Returns the versions applied by this call (empty if already current)."""
    migrations = discover(directory)

    # Taken before anything else, including creating the bookkeeping table.
    # `CREATE TABLE IF NOT EXISTS` is not atomic across concurrent sessions —
    # two runners can both find it missing and both try to create it, and the
    # loser gets a duplicate-key error on pg_type rather than a clean no-op.
    # An advisory lock needs no table of its own, so it can and must come first.
    await conn.execute("SELECT pg_advisory_lock($1)", _ADVISORY_LOCK_KEY)
    try:
        await conn.execute(_SCHEMA_MIGRATIONS_DDL)

        applied = await _applied(conn)
        _verify_checksums(migrations, applied)

        newly_applied: list[str] = []
        for migration in migrations:
            if migration.version in applied:
                continue
            async with conn.transaction():
                await conn.execute(migration.sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (version, checksum) VALUES ($1, $2)",
                    migration.version,
                    migration.checksum,
                )
            logger.info("applied migration %s", migration.version)
            newly_applied.append(migration.version)
        return newly_applied
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_KEY)


async def status(conn: asyncpg.Connection, directory: Path = MIGRATIONS_DIR) -> list[tuple[str, bool]]:
    """Read-only: reporting status must never mutate the database, so an
    absent bookkeeping table means "nothing applied yet", not "create it"."""
    table_exists = await conn.fetchval("SELECT to_regclass('schema_migrations') IS NOT NULL")
    applied = await _applied(conn) if table_exists else {}
    return [(m.version, m.version in applied) for m in discover(directory)]


async def _run(show_status: bool) -> int:
    conn = await asyncpg.connect(settings.database_url)
    try:
        if show_status:
            for version, is_applied in await status(conn):
                print(f"  [{'x' if is_applied else ' '}] {version}")
            return 0

        applied = await apply_pending(conn)
        if applied:
            for version in applied:
                print(f"  applied {version}")
        else:
            print("  already up to date")
        return 0
    finally:
        await conn.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return asyncio.run(_run(show_status="--status" in sys.argv[1:]))
    except MigrationError as exc:
        print(f"migration failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
