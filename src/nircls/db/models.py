from __future__ import annotations

from sqlalchemy import JSON, Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from .database import Base


class Role(Base):
    __tablename__ = "roles"

    role_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)

    users = relationship("User", secondary="user_roles", back_populates="roles")


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(128), nullable=False)

    roles = relationship("Role", secondary="user_roles", back_populates="users")


class Variavel(Base):
    __tablename__ = "variaveis"

    var_id = Column(Integer, primary_key=True, autoincrement=True)
    nome = Column(String(100), nullable=False)


class UserRole(Base):
    __tablename__ = "user_roles"

    user_id = Column(Integer, ForeignKey("users.user_id"), primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.role_id"), primary_key=True)


class SampleMetadata(Base):
    __tablename__ = "l"

    sample_id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(100), nullable=True)
    description = Column(String(255), nullable=True)
    metadata_json = Column("metadata", JSON, nullable=True)

    features = relationship("SampleFeatures", back_populates="sample", uselist=False)
    targets = relationship("SampleTarget", back_populates="sample", cascade="all, delete-orphan")


class SampleFeatures(Base):
    __tablename__ = "x"
    __table_args__ = (UniqueConstraint("sample_id", name="uq_x_sample"),)

    x_id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(Integer, ForeignKey("l.sample_id"), nullable=False)
    values = Column(JSON, nullable=False)

    sample = relationship("SampleMetadata", back_populates="features")


class SampleTarget(Base):
    __tablename__ = "y"

    y_id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(Integer, ForeignKey("l.sample_id"), nullable=False)
    target_name = Column(String(100), nullable=False)
    target_value = Column(String(255), nullable=False)

    sample = relationship("SampleMetadata", back_populates="targets")
