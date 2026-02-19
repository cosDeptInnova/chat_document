from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, ForeignKey, Text, DateTime,
    Enum, JSON, Float, Numeric, Table
)
from sqlalchemy.orm import relationship
from config.database import Base

# Association table for User <-> Department (many-to-many)
user_departments = Table(
    'user_departments', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('department_id', Integer, ForeignKey('departments.id'), primary_key=True)
)

class Role(Base):
    __tablename__ = 'roles'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=False)

    users = relationship('User', back_populates='role')

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role_id = Column(Integer, ForeignKey('roles.id'), nullable=False)
    user_directory = Column(String(255), nullable=False)
    faiss_index_path = Column(String(255), nullable=True)
    vectorizer_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    role = relationship('Role', back_populates='users')
    departments = relationship(
        'Department', secondary=user_departments,
        back_populates='users'
    )
    conversations = relationship(
        'Conversation', back_populates='user',
        cascade='all, delete-orphan'
    )
    sessions = relationship('Session', back_populates='user', cascade='all, delete-orphan')
    audit_logs = relationship('AuditLog', back_populates='user', cascade='all, delete-orphan')

class Department(Base):
    __tablename__ = 'departments'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    department_directory = Column(String(255), nullable=False)
    faiss_index_path = Column(String(255), nullable=True)
    vectorizer_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    users = relationship(
        'User', secondary=user_departments,
        back_populates='departments'
    )

class File(Base):
    __tablename__ = 'files'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    department_id = Column(Integer, ForeignKey('departments.id'), nullable=True)
    file_path = Column(String(255), nullable=False)
    file_name = Column(String(255), nullable=False)
    permission = Column(
        Enum('READ', 'WRITE', 'NONE', name='file_permission'),
        nullable=False
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship('User')
    department = relationship('Department')

class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    conversation_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_deleted = Column(Boolean, default=False)
    is_favorite = Column(Boolean, default=False)

    user = relationship('User', back_populates='conversations')
    messages = relationship(
        'Message', back_populates='conversation',
        cascade='all, delete-orphan'
    )
    meta = relationship(
        'ConversationMeta', back_populates='conversation',
        uselist=False, cascade='all, delete-orphan'
    )

class Message(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    sender = Column(
        Enum('USER', 'SYSTEM', 'BOT', name='message_sender'),
        nullable=False
    )
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_liked = Column(Boolean, nullable=True, default=None)

    conversation = relationship('Conversation', back_populates='messages')
    attachments = relationship(
        'Attachment', back_populates='message',
        cascade='all, delete-orphan'
    )
    usage_logs = relationship(
        'UsageLog', back_populates='message',
        cascade='all, delete-orphan'
    )

class ConversationMeta(Base):
    __tablename__ = 'conversation_meta'

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(
        Integer, ForeignKey('conversations.id'),
        unique=True, nullable=False
    )
    system_prompt = Column(Text, nullable=True)
    temperature = Column(Float, default=0.7, nullable=False)
    max_tokens = Column(Integer, default=1024, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    conversation = relationship('Conversation', back_populates='meta')

class Attachment(Base):
    __tablename__ = 'attachments'

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey('messages.id'), nullable=False)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    message = relationship('Message', back_populates='attachments')
    file = relationship('File')

class UsageLog(Base):
    __tablename__ = 'usage_logs'

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey('messages.id'), nullable=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    model_name = Column(String(100), nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    cost = Column(Numeric(10, 6), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    message = relationship('Message', back_populates='usage_logs')
    conversation = relationship('Conversation')

class Session(Base):
    __tablename__ = 'sessions'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    session_token = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    user = relationship('User', back_populates='sessions')

class AuditLog(Base):
    __tablename__ = 'audit_logs'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    entity_name = Column(String(100), nullable=False)
    entity_id = Column(Integer, nullable=False)
    action = Column(
        Enum('CREATE', 'UPDATE', 'DELETE', name='audit_action'),
        nullable=False
    )
    old_data = Column(JSON, nullable=True)
    new_data = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship('User', back_populates='audit_logs')

class ConversationHistory(Base):
    __tablename__ = 'conversations_history'

    history_id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)
    conversation_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    operation = Column(
        Enum('INSERT', 'UPDATE', 'DELETE', name='conv_history_op'),
        nullable=False
    )

# Alias for backward compatibility
UserDepartment = user_departments
