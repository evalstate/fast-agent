# src/database/config.py
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict

class Settings(BaseSettings):
    # full URL override (optional)
    database_url: str | None = None

    # forbid any other env vars aside from those explicitly declared
    model_config = ConfigDict(extra="forbid")

    # these will default if the corresponding env var is missing
    DB_HOST:     str = Field("34.47.187.64", env="DB_HOST")
    DB_NAME:     str = Field("postgres", env="DB_NAME")
    DB_PORT:     str = Field("5432",denv="DB_PORT")

    DB_USER:     str = Field("admin",env="DB_USER")
    DB_PASSWORD: str = Field("admin",env="DB_PASSWORD")

    @property
    def DATABASE_URL(self) -> str:
        return (
            self.database_url
            or f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
              f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

settings = Settings()
