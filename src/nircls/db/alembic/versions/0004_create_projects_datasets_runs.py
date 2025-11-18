"""Create projects, datasets, and runs tables"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0004_create_projects_datasets_runs"
down_revision = "0003_make_l_metadata_optional"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("project_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.user_id"), nullable=True),
    )

    op.create_table(
        "datasets",
        sa.Column("dataset_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=True),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.Column(
            "project_id", sa.Integer(), sa.ForeignKey("projects.project_id"), nullable=True
        ),
        sa.Column(
            "created_by_id", sa.Integer(), sa.ForeignKey("users.user_id"), nullable=True
        ),
        sa.Column("storage_path", sa.String(length=500), nullable=True),
        sa.UniqueConstraint("name", "version", "project_id", name="uq_dataset_version"),
    )
    op.create_index("ix_datasets_project_id", "datasets", ["project_id"])
    op.create_index("ix_datasets_created_by_id", "datasets", ["created_by_id"])

    op.create_table(
        "runs",
        sa.Column("run_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_uid", sa.String(length=36), nullable=True, unique=True),
        sa.Column(
            "project_id", sa.Integer(), sa.ForeignKey("projects.project_id"), nullable=True
        ),
        sa.Column(
            "dataset_id", sa.Integer(), sa.ForeignKey("datasets.dataset_id"), nullable=True
        ),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.user_id"), nullable=True),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="running"),
        sa.Column("artifacts", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )
    op.create_index("ix_runs_project_id", "runs", ["project_id"])
    op.create_index("ix_runs_dataset_id", "runs", ["dataset_id"])
    op.create_index("ix_runs_user_id", "runs", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_runs_user_id", table_name="runs")
    op.drop_index("ix_runs_dataset_id", table_name="runs")
    op.drop_index("ix_runs_project_id", table_name="runs")
    op.drop_table("runs")

    op.drop_index("ix_datasets_created_by_id", table_name="datasets")
    op.drop_index("ix_datasets_project_id", table_name="datasets")
    op.drop_table("datasets")

    op.drop_table("projects")
