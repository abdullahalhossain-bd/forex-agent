# alembic/README.md
# ============================================================
# Database Migrations (Alembic)
# ============================================================
#
# The forex-agent project used to manage schema changes via raw
# ALTER TABLE statements embedded in database/db.py's _migrate_*
# methods. That worked for additive changes but had two problems:
#
#   1. No rollback path — once a column was added, you couldn't
#      cleanly revert a bad migration.
#   2. No version tracking — you couldn't tell which schema version
#      a given .db file was at, so reproducing a bug on a fresh DB
#      vs. an old DB was a guessing game.
#
# Alembic fixes both. The existing _migrate_* methods in db.py are
# left in place (defensive — they're idempotent), but new schema
# changes should go through Alembic.
#
# ## Common commands
#
#   # Apply all pending migrations (run after git pull):
#   alembic upgrade head
#
#   # Create a new migration after editing the schema:
#   alembic revision --autogenerate -m "add foo column to trades"
#
#   # Show current DB revision:
#   alembic current
#
#   # Show migration history:
#   alembic history --verbose
#
#   # Roll back one migration:
#   alembic downgrade -1
#
#   # Roll back to a specific revision:
#   alembic downgrade 0001_initial
#
# ## How to add a new column
#
#   1. Edit database/db.py to include the new column in CREATE TABLE
#      (so fresh databases get it).
#   2. Generate a migration:
#        alembic revision -m "add foo to trades"
#   3. Edit alembic/versions/<new_file>.py to add the ALTER TABLE:
#        def upgrade():
#            op.add_column('trades', sa.Column('foo', sa.String(50)))
#        def downgrade():
#            op.drop_column('trades', 'foo')
#   4. Apply it:
#        alembic upgrade head
#
# ## Production database
#
# Point Alembic at a different DB via the env var:
#
#   SQLALCHEMY_DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/forex \
#     alembic upgrade head
#
# This is the cleanest way to migrate from SQLite → Postgres when
# the project outgrows SQLite.
