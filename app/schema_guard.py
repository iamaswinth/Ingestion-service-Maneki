"""Shared policy for "should this process create its own tables?".

Three modules own a slice of the schema (app/storage.py, app/ingestion/store.py,
app/salescript/store.py) and each used to apply its own DDL on first use. That
is convenient locally and wrong in production: it requires the application's
own database role to keep CREATE rights forever, and it hides a failed or
skipped deploy step behind a table that quietly reappears.

So in production the schema is owned by `python -m app.migrate`, and these
modules only verify. The check exists so a missed migration fails at startup
with a message naming the problem, rather than as an UndefinedTableError from
somewhere deep inside an unrelated query.
"""

from .config import settings


class SchemaNotMigrated(RuntimeError):
    pass


def auto_create_enabled() -> bool:
    return settings.environment.strip().lower() not in ("production", "staging")


async def verify_tables(conn, tables: tuple[str, ...]) -> None:
    missing = [
        table
        for table in tables
        if not await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", table)
    ]
    if missing:
        raise SchemaNotMigrated(
            f"missing table(s): {', '.join(missing)}. Run `python -m app.migrate` "
            "as part of the deploy; the application does not create its own "
            "tables outside development."
        )
