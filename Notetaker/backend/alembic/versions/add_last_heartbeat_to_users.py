"""add_last_heartbeat_to_users

Revision ID: add_last_heartbeat
Revises: 9225cea883fe
Create Date: 2026-01-20 14:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_last_heartbeat'
down_revision = '9225cea883fe'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Agregar campo last_heartbeat a la tabla users
    op.add_column(
        'users',
        sa.Column('last_heartbeat', sa.DateTime(), nullable=True)
    )
    # Crear índice para mejorar las consultas de usuarios online
    op.create_index('ix_users_last_heartbeat', 'users', ['last_heartbeat'])


def downgrade() -> None:
    # Eliminar índice
    op.drop_index('ix_users_last_heartbeat', table_name='users')
    # Eliminar campo last_heartbeat
    op.drop_column('users', 'last_heartbeat')
