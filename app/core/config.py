"""
Application configuration settings
"""

import os
from typing import List


class Settings:
    # Project Info
    PROJECT_NAME: str = "Dentsu News Platform API"
    PROJECT_DESCRIPTION: str = "AI-powered news platform for Dentsu technical interview"
    VERSION: str = "1.0.0"
    
    # Environment
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://news-dentsu-frontend.vercel.app",
        "*"  # En producción, especificar dominios exactos
    ]
    
    # External APIs
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))


settings = Settings()
