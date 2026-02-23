"""add_password_fields_to_users

Revision ID: add_password_fields
Revises: 06f584571d4d
Create Date: 2025-01-XX XX:XX:XX.XXXXXX

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_password_fields'
down_revision = '06f584571d4d'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Agregar campos de contraseña a la tabla users
    op.add_column(
        'users',
        sa.Column('hashed_password', sa.String(), nullable=True)
    )
    op.add_column(
        'users',
        sa.Column('password_reset_token', sa.String(), nullable=True)
    )
    op.add_column(
        'users',
        sa.Column('password_reset_expires', sa.DateTime(), nullable=True)
    )
    
    # Crear índice para password_reset_token para búsquedas rápidas
    op.create_index(
        'ix_users_password_reset_token',
        'users',
        ['password_reset_token'],
        unique=False
    )


def downgrade() -> None:
    # Eliminar índice
    op.drop_index('ix_users_password_reset_token', table_name='users')
    
    # Eliminar columnas
    op.drop_column('users', 'password_reset_expires')
    op.drop_column('users', 'password_reset_token')
    op.drop_column('users', 'hashed_password')

