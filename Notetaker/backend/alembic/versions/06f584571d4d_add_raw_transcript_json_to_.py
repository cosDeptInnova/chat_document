"""add_raw_transcript_json_to_transcriptions

Revision ID: 06f584571d4d
Revises: 
Create Date: 2025-12-19 11:09:13.390726

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '06f584571d4d'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Agregar columna raw_transcript_json a la tabla transcriptions
    op.add_column(
        'transcriptions',
        sa.Column('raw_transcript_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True)
    )


def downgrade() -> None:
    # Eliminar columna raw_transcript_json
    op.drop_column('transcriptions', 'raw_transcript_json')

