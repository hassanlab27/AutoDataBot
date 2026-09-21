from typing import List, Dict, Any
from pydantic import BaseModel, Field

class RepairAction(BaseModel):
    action_type: str  # e.g., "encoding_detection", "delimiter_detection", "duplicate_columns", "unnamed_index"
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)

class IngestionReport(BaseModel):
    original_filename: str
    file_size_bytes: int
    encoding_used: str
    delimiter_used: str
    total_raw_rows: int
    total_raw_columns: int
    skipped_bad_lines: int = 0
    actions: List[RepairAction] = Field(default_factory=list)

    def add_action(self, action_type: str, description: str, details: Dict[str, Any] = None):
        self.actions.append(
            RepairAction(
                action_type=action_type,
                description=description,
                details=details or {}
            )
        )
