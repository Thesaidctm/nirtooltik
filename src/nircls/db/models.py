from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from .database import Base


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, server_default=sa.func.now())

    users = relationship("User", back_populates="organization")
    projects = relationship("Project", back_populates="organization")
    datasets = relationship("Dataset", back_populates="organization")
    runs = relationship("Run", back_populates="organization")
    instruments = relationship("Instrument", back_populates="organization")


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
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)

    roles = relationship("Role", secondary="user_roles", back_populates="users")
    projects = relationship("Project", back_populates="owner")
    datasets = relationship("Dataset", back_populates="created_by")
    runs = relationship("Run", back_populates="user")
    organization = relationship("Organization", back_populates="users")


class Variavel(Base):
    __tablename__ = "variaveis"

    var_id = Column(Integer, primary_key=True, autoincrement=True)
    nome = Column(String(100), nullable=False)


class UserRole(Base):
    __tablename__ = "user_roles"

    user_id = Column(Integer, ForeignKey("users.user_id"), primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.role_id"), primary_key=True)


class Project(Base):
    __tablename__ = "projects"

    project_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(String(500), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)

    owner = relationship("User", back_populates="projects")
    datasets = relationship(
        "Dataset", back_populates="project", cascade="all, delete-orphan"
    )
    runs = relationship("Run", back_populates="project")
    organization = relationship("Organization", back_populates="projects")


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        UniqueConstraint("name", "version", "project_id", name="uq_dataset_version"),
    )

    dataset_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=True)
    description = Column(String(500), nullable=True)
    project_id = Column(Integer, ForeignKey("projects.project_id"), nullable=True)
    created_by_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    storage_path = Column(String(500), nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=True)

    project = relationship("Project", back_populates="datasets")
    created_by = relationship("User", back_populates="datasets")
    runs = relationship("Run", back_populates="dataset")
    organization = relationship("Organization", back_populates="datasets")
    instrument = relationship("Instrument", back_populates="datasets")


class Run(Base):
    __tablename__ = "runs"

    run_id = Column(Integer, primary_key=True, autoincrement=True)
    run_uid = Column(String(36), unique=True, nullable=True)
    project_id = Column(Integer, ForeignKey("projects.project_id"), nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.dataset_id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)
    config = Column(JSON, nullable=False)
    status = Column(String(50), nullable=False, default="running")
    artifacts = Column(JSON, nullable=True)
    error_message = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    project = relationship("Project", back_populates="runs")
    dataset = relationship("Dataset", back_populates="runs")
    user = relationship("User", back_populates="runs")
    models = relationship("ModelRegistry", back_populates="run")
    organization = relationship("Organization", back_populates="runs")


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)

    run_id = Column(Integer, ForeignKey("runs.run_id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.dataset_id"), nullable=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=True)
    target = Column(String, nullable=True)

    status = Column(String, nullable=False, default="candidate")
    created_at = Column(DateTime, server_default=sa.func.now())
    updated_at = Column(DateTime, server_default=sa.func.now(), onupdate=sa.func.now())
    eval_history = Column(JSON, nullable=True)

    run = relationship("Run", back_populates="models")
    dataset = relationship("Dataset")
    instrument = relationship("Instrument", back_populates="models")


class Instrument(Base):
    __tablename__ = "instruments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    code = Column(String, unique=True, nullable=False)
    vendor = Column(String, nullable=True)
    model = Column(String, nullable=True)
    serial_number = Column(String, nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)

    created_at = Column(DateTime, server_default=sa.func.now())
    updated_at = Column(
        DateTime, server_default=sa.func.now(), onupdate=sa.func.now()
    )

    models = relationship("ModelRegistry", back_populates="instrument", lazy="selectin")
    datasets = relationship("Dataset", back_populates="instrument", lazy="selectin")
    organization = relationship("Organization", back_populates="instruments")


class CalibrationTransfer(Base):
    __tablename__ = "calibration_transfers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)

    master_model_id = Column(Integer, ForeignKey("model_registry.id"), nullable=False)
    master_instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    slave_instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)

    method = Column(String, nullable=False, default="ds")
    artifacts = Column(JSON, nullable=True)
    metadata_json = Column("metadata", JSON, nullable=True)

    standard_dataset_master_id = Column(Integer, ForeignKey("datasets.dataset_id"), nullable=True)
    standard_dataset_slave_id = Column(Integer, ForeignKey("datasets.dataset_id"), nullable=True)

    created_at = Column(DateTime, server_default=sa.func.now())
    updated_at = Column(
        DateTime, server_default=sa.func.now(), onupdate=sa.func.now()
    )

    master_model = relationship("ModelRegistry")
    master_instrument = relationship(
        "Instrument", foreign_keys=[master_instrument_id]
    )
    slave_instrument = relationship(
        "Instrument", foreign_keys=[slave_instrument_id]
    )
    standard_dataset_master = relationship(
        "Dataset", foreign_keys=[standard_dataset_master_id]
    )
    standard_dataset_slave = relationship(
        "Dataset", foreign_keys=[standard_dataset_slave_id]
    )


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
