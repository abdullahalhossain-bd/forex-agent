# alembic/env.py — Alembic migration environment
# ============================================================
# Reads DB URL from env (SQLALCHEMY_DATABASE_URL) or falls back
# to the SQLite path used by database/db.py.
#
# Migrations are written SQLAlchemy-native (not autogenerate by
# default) so they stay readable and portable across SQLite/PG/MySQL.
# ============================================================

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool, create_engine

# Make sure project root is on sys.path so config.db_path etc. import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load alembic.ini config
config = context.config

# Resolve DB URL — env var overrides alembic.ini
env_url = os.getenv("SQLALCHEMY_DATABASE_URL", "").strip()
if env_url:
    config.set_main_option("sqlalchemy.url", env_url)
else:
    # Default: SQLite at database/trader.db (matches database/db.py)
    db_path = PROJECT_ROOT / "database" / "trader.db"
    config.set_main_option(
        "sqlalchemy.url",
        f"sqlite:///{db_path}",
    )

# Configure Python logging from alembic.ini
if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name)
    except Exception:
        pass  # don't fail migrations over logging setup issues


# ── Target metadata ──────────────────────────────────────────
# For autogenerate support, we'd set target_metadata = Base.metadata
# here. The forex-agent project uses raw SQL DDL (not SQLAlchemy ORM
# models), so autogenerate isn't useful. Migrations are hand-written.
target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL to stdout."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — connect to the DB and apply.

    Uses a fresh engine with autocommit disabled (the default in
    SQLAlchemy 2.x). The migration runs inside context.begin_transaction()
    which commits on success, rolls back on failure.
    """
    url = config.get_main_option("sqlalchemy.url")

    # Special-case SQLite — use a simple file engine, no pool
    if url.startswith("sqlite"):
        engine = create_engine(url, poolclass=pool.NullPool)
    else:
        engine = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    with engine.connect() as connection:
        # For SQLite + DDL: ensure the connection is in autocommit mode
        # so CREATE TABLE persists. SQLAlchemy 2.x defaults to
        # transactional mode; for DDL-only migrations we want autocommit.
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=url.startswith("sqlite"),  # batch mode for SQLite ALTER
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
