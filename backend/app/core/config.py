from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "AutoDataBot API"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # Upload and dataset policies
    MAX_UPLOAD_SIZE_MB: int = 100  # Default 100MB limit
    DEFAULT_PREVIEW_ROWS: int = 20
    MAX_PREVIEW_ROWS: int = 100
    
    # Storage paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"
    
    # Thresholds for heuristic detection
    HIGH_CARDINALITY_RATIO_THRESHOLD: float = 0.50
    SUSPICIOUS_ID_RATIO_THRESHOLD: float = 0.95
    CONSTANT_COLUMN_THRESHOLD: int = 1

    # Phase 2: EDA & Visualization Limits
    EDA_MAX_NUMERIC_DISTRIBUTION_COLUMNS: int = 10
    EDA_MAX_CATEGORICAL_CHART_COLUMNS: int = 10
    EDA_MAX_CATEGORIES_PER_CHART: int = 10
    EDA_MAX_CORRELATION_COLUMNS: int = 20
    EDA_MAX_SCATTER_POINTS: int = 2000
    EDA_TOP_CORRELATION_PAIRS: int = 10
    EDA_OUTLIER_IQR_MULTIPLIER: float = 1.5
    EDA_STRONG_CORRELATION_THRESHOLD: float = 0.5
    EDA_RANDOM_SEED: int = 42

    model_config = {
        "env_prefix": "AUTODATABOT_",
        "case_sensitive": False
    }

settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
