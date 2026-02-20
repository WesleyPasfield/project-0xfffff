"""Add generated_skills table.

Stores agent skills generated from workshop artifacts (optimized prompt,
aligned judge memory, evaluated traces). Skills are scoped per workshop
and written to Lakebase for participants to integrate into their agents.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0014_add_generated_skills"
down_revision = "0013_add_optimized_uri_to_optimization_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "generated_skills",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "workshop_id",
            sa.String(),
            sa.ForeignKey("workshops.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("generation_model", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("workshop_id", "name", name="uq_generated_skills_workshop_name"),
    )
    op.create_index(
        "ix_generated_skills_workshop_id",
        "generated_skills",
        ["workshop_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_generated_skills_workshop_id",
        table_name="generated_skills",
    )
    op.drop_table("generated_skills")
