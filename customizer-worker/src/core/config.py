from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database
    hf_token: str

    # App
    port: int = 8000

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
