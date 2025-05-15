# src/database/config.py
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    # Add explicit field for database_url
    database_url: str | None = None
    
    model_config = ConfigDict(extra="forbid")

    DB_HOST:     str = "34.47.187.64"
    DB_NAME:     str = "postgres"
    DB_USER:     str = "admin"
    DB_PASSWORD: str = "admin"
    DB_PORT:     str = "5432"

    @property
    def DATABASE_URL(self) -> str:
        return (
            # if the env var DATABASE_URL exists, use it
            self.database_url
            or f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
              f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

settings = Settings()