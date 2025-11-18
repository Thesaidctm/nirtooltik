"""Create X, Y, L dataset tables"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0002_create_xy_l_tables"
down_revision = "0001_create_core_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "l",
        sa.Column("sample_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("external_id", sa.String(length=100), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
    )

    op.create_table(
        "x",
        sa.Column("x_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("sample_id", sa.Integer(), sa.ForeignKey("l.sample_id"), nullable=False),
        sa.Column("values", sa.JSON(), nullable=False),
        sa.UniqueConstraint("sample_id", name="uq_x_sample"),
    )
    op.create_index("ix_x_sample_id", "x", ["sample_id"])

    op.create_table(
        "y",
        sa.Column("y_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("sample_id", sa.Integer(), sa.ForeignKey("l.sample_id"), nullable=False),
        sa.Column("target_name", sa.String(length=100), nullable=False),
        sa.Column("target_value", sa.String(length=255), nullable=False),
    )
    op.create_index("ix_y_sample_id", "y", ["sample_id"])


def downgrade() -> None:
    op.drop_index("ix_y_sample_id", table_name="y")
    op.drop_table("y")
    op.drop_index("ix_x_sample_id", table_name="x")
    op.drop_table("x")
    op.drop_table("l")
