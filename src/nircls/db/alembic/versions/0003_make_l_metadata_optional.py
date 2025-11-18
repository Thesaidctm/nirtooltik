"""Make L metadata optional"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0003_make_l_metadata_optional"
down_revision = "0002_create_xy_l_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "l",
        "external_id",
        existing_type=sa.String(length=100),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "l",
        "external_id",
        existing_type=sa.String(length=100),
        nullable=False,
    )
