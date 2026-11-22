import os
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/jobmatcher"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    COHERE_API_KEY: str = ""
    
    HOST: str = "0.0.0.0"
    PORT: int = int(os.environ.get("PORT", 8000))
    APP_ENV: str = "development"

    @property
    def sanitized_database_url(self) -> str:
        """
        Normalizes connection strings from Neon, Supabase, or Railway:
        1. Converts 'postgres://' or 'postgresql://' to 'postgresql+asyncpg://'
        2. Strips 'sslmode' query params which conflict with asyncpg kwargs.
        """
        url = self.DATABASE_URL.strip()
        if url.startswith("postgres://"):
            url = "postgresql+asyncpg://" + url[len("postgres://"):]
        elif url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
            url = "postgresql+asyncpg://" + url[len("postgresql://"):]

        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        if "sslmode" in query_params:
            del query_params["sslmode"]
            new_query = urlencode(query_params, doseq=True)
            url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment,
            ))
        return url

    @property
    def is_cloud_db(self) -> bool:
        """Detects if connecting to hosted Supabase, Neon, or cloud Postgres requiring SSL."""
        lower = self.DATABASE_URL.lower()
        return any(k in lower for k in ["supabase.co", "neon.tech", "pooler.supabase", "render.com", "sslmode=require"])


settings = Settings()
