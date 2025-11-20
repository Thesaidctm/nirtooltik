#!/usr/bin/env python
from __future__ import annotations

"""Utility to create or reset an admin user in the configured database."""

import argparse

from nircls.core.services.users import ensure_admin_user
from nircls.db import Base, SessionLocal, engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update an admin user with the provided credentials.",
    )
    parser.add_argument(
        "--username",
        default="admin",
        help="Username for the admin account (default: admin)",
    )
    parser.add_argument(
        "--password",
        default="admin",
        help="Password to assign to the admin account (default: admin)",
    )
    parser.add_argument(
        "--email",
        default="admin@example.com",
        help="Email for the admin account (default: admin@example.com)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure tables exist before attempting to insert data.
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        user = ensure_admin_user(
            db,
            username=args.username,
            email=args.email,
            password=args.password,
        )
    finally:
        db.close()

    print(
        "Admin user ensured:",
        f"id={user.user_id}",
        f"username={user.username}",
        f"email={user.email}",
    )


if __name__ == "__main__":
    main()
