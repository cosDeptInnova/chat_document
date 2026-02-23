"""add_must_change_password_to_users

Revision ID: 9225cea883fe
Revises: add_password_fields
Create Date: 2025-12-20 18:17:17.210566

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9225cea883fe'
down_revision = 'add_password_fields'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Agregar campo must_change_password a la tabla users
    op.add_column(
        'users',
        sa.Column('must_change_password', sa.Boolean(), nullable=False, server_default='false')
    )


def downgrade() -> None:
    # Eliminar campo must_change_password
    op.drop_column('users', 'must_change_password')

