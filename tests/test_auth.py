from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from nircls.auth import SESSION_COOKIE_NAME, SESSIONS, app, get_db, hash_password
from nircls.db.models import Base, User


def setup_test_app():
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestingSessionLocal


def test_login_and_logout():
    TestingSessionLocal = setup_test_app()
    with TestingSessionLocal() as db:
        user = User(
            username="alice",
            email="alice@example.com",
            hashed_password=hash_password("secret"),
        )
        db.add(user)
        db.commit()

    client = TestClient(app)

    # Successful login
    resp = client.post("/login", json={"username": "alice", "password": "secret"})
    assert resp.status_code == 200
    token = resp.cookies.get(SESSION_COOKIE_NAME)
    assert token in SESSIONS

    # Failed login
    resp = client.post("/login", json={"username": "alice", "password": "wrong"})
    assert resp.status_code == 401

    # Logout should invalidate session
    resp = client.post("/logout", cookies={SESSION_COOKIE_NAME: token})
    assert resp.status_code == 200
    assert token not in SESSIONS
    assert SESSION_COOKIE_NAME not in client.cookies
