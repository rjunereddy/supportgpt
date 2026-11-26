import os
from typing import Optional
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

ENV_PATH = os.path.join(os.path.dirname(__file__), '..', '.env')

class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "mock")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "mock")

def reload_env():
    """Reloads environment variables from .env file."""
    if os.path.exists(ENV_PATH):
        try:
            with open(ENV_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        os.environ[k.strip()] = v.strip()
        except Exception as e:
            print(f"Error loading .env: {e}")

def get_settings() -> Settings:
    reload_env()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "mock"),
        groq_api_key=os.getenv("GROQ_API_KEY", "mock")
    )

def update_keys(openai_api_key: Optional[str] = None, groq_api_key: Optional[str] = None, persist: bool = True):
    """Updates active environment variables and optionally writes them to .env."""
    if openai_api_key is not None:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if groq_api_key is not None:
        os.environ["GROQ_API_KEY"] = groq_api_key

    if persist:
        existing_lines = {}
        if os.path.exists(ENV_PATH):
            try:
                with open(ENV_PATH, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            existing_lines[k.strip()] = v.strip()
            except Exception:
                pass

        if openai_api_key is not None:
            existing_lines["OPENAI_API_KEY"] = openai_api_key
        if groq_api_key is not None:
            existing_lines["GROQ_API_KEY"] = groq_api_key

        try:
            with open(ENV_PATH, 'w', encoding='utf-8') as f:
                for k, v in existing_lines.items():
                    f.write(f"{k}={v}\n")
        except Exception as e:
            print(f"Failed to persist keys to .env: {e}")
