from typing import Optional, Literal
from pydantic import BaseModel, Field
import os

AUTOML_MODE_TIME_LIMITS = {
    "quick": 60,
    "standard": 300,
    "extended": 900
}

TrainingMode = Literal["quick", "standard", "extended", "custom"]
EnginePreference = Literal["all", "autogluon", "flaml", "sklearn_gbdt"]

class AutoMLConfig(BaseModel):
    dataset_id: str
    target: str
    problem_type: str = Field(description="binary_classification, multiclass_classification, or regression")
    training_mode: TrainingMode = "standard"
    time_limit: Optional[int] = Field(default=None, description="Time limit in seconds; overrides training_mode if provided")
    primary_metric: str = Field(default="auto", description="Primary metric for model selection, or 'auto'")
    random_state: int = Field(default=42, description="Seed for reproducible training")
    max_cpus: Optional[int] = Field(default=None, description="Max CPU cores to utilize; defaults to min(os.cpu_count(), 8)")
    presets: str = Field(default="medium_quality", description="AutoGluon preset level")
    engine_preference: EnginePreference = Field(default="all", description="Allowed engines for this run")

    def get_effective_time_limit(self) -> int:
        if self.time_limit is not None and self.time_limit > 0:
            return self.time_limit
        return AUTOML_MODE_TIME_LIMITS.get(self.training_mode, 300)

    def get_effective_cpus(self) -> int:
        available = os.cpu_count() or 2
        if self.max_cpus is not None and self.max_cpus > 0:
            return min(self.max_cpus, available)
        return max(1, min(available, 8))
