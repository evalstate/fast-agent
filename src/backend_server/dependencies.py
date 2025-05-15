from typing import Generator

from ..database.database import SessionLocal


def get_db() -> Generator:
    """
    Dependency for database session.
    
    Yields:
        Database session that is closed after use
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()