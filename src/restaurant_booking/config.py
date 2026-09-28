from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Settings:
    groq_api_key: str
    groq_model: str
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @classmethod
    def from_environment(cls) -> Settings:
        load_dotenv(PROJECT_ROOT / ".env", override=False)

        values = {
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY", "").strip(),
            "LANGFUSE_PUBLIC_KEY": os.getenv("LANGFUSE_PUBLIC_KEY", "").strip(),
            "LANGFUSE_SECRET_KEY": os.getenv("LANGFUSE_SECRET_KEY", "").strip(),
        }
        if not values["GROQ_API_KEY"]:
            raise ValueError("Missing required environment variables: GROQ_API_KEY")

        return cls(
            groq_api_key=values["GROQ_API_KEY"],
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
            langfuse_public_key=values["LANGFUSE_PUBLIC_KEY"],
            langfuse_secret_key=values["LANGFUSE_SECRET_KEY"],
            langfuse_base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").strip(),
        )
