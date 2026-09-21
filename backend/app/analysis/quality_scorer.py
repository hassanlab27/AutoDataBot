import re
from typing import List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field
from app.validation.schema_inspector import DatasetStructureSummary, ColumnSummary

class SuspiciousColumn(BaseModel):
    column_name: str
    issue_type: str  # "id_like", "timestamp_like", "constant", "high_cardinality", "high_missingness"
    severity: str    # "warning", "info"
    description: str

class QualityCheck(BaseModel):
    name: str
    status: str       # "pass", "warning", "fail"
    summary: str
    description: str
    affected_columns: List[str] = Field(default_factory=list)

class DatasetQualityReport(BaseModel):
    quality_score: int
    rating: str       # "Excellent", "Good", "Fair", "Needs Attention"
    checks: List[QualityCheck]
    suspicious_columns: List[SuspiciousColumn]
    score_breakdown: Dict[str, int]

def detect_suspicious_columns(summary: DatasetStructureSummary) -> List[SuspiciousColumn]:
    """Flag potentially suspicious features without altering the dataset."""
    suspicious = []
    id_name_pattern = re.compile(r"(^|_)(id|uuid|pk|key|index|account|customer_id|user_id)($|_)", re.IGNORECASE)
    date_name_pattern = re.compile(r"(^|_)(date|time|timestamp|datetime|year|month|day)($|_)", re.IGNORECASE)

    for col in summary.columns:
        # 1. Constant Column
        if col.is_constant:
            suspicious.append(
                SuspiciousColumn(
                    column_name=col.name,
                    issue_type="constant",
                    severity="warning",
                    description=(
                        f"Column '{col.name}' contains only 1 unique value across all rows. "
                        "It provides zero variance and no predictive information."
                    )
                )
            )

        # 2. ID-like Column
        is_id_named = bool(id_name_pattern.search(col.name))
        is_near_unique = (col.unique_percentage >= 95.0 and summary.row_count > 30)

        if (is_id_named and col.unique_percentage > 50.0) or (is_near_unique and col.inferred_dtype in ("integer", "categorical", "text")):
            suspicious.append(
                SuspiciousColumn(
                    column_name=col.name,
                    issue_type="id_like",
                    severity="warning",
                    description=(
                        f"Column '{col.name}' has {col.unique_percentage}% unique values and ID-like characteristics. "
                        "This likely represents an entity identifier rather than a generalizing feature."
                    )
                )
            )

        # 3. High Cardinality Categorical
        if col.is_high_cardinality and col.inferred_dtype in ("categorical", "text"):
            suspicious.append(
                SuspiciousColumn(
                    column_name=col.name,
                    issue_type="high_cardinality",
                    severity="warning",
                    description=(
                        f"Column '{col.name}' has {col.unique_count} distinct categories ({col.unique_percentage}% of rows). "
                        "High cardinality can cause overfitting or sparse encodings in tree-based and linear models."
                    )
                )
            )

        # 4. Timestamp-like Column
        if col.inferred_dtype == "datetime-like" or bool(date_name_pattern.search(col.name)):
            suspicious.append(
                SuspiciousColumn(
                    column_name=col.name,
                    issue_type="timestamp_like",
                    severity="info",
                    description=(
                        f"Column '{col.name}' exhibits temporal or date characteristics. "
                        "Special date feature extraction (year, month, day, day-of-week) may be beneficial."
                    )
                )
            )

        # 5. Extreme Missingness
        if col.missing_percentage >= 50.0:
            suspicious.append(
                SuspiciousColumn(
                    column_name=col.name,
                    issue_type="high_missingness",
                    severity="warning",
                    description=(
                        f"Column '{col.name}' is missing {col.missing_percentage}% of its values. "
                        "Imputation may introduce noise or require a missing indicator feature."
                    )
                )
            )

    return suspicious


def calculate_quality_score(summary: DatasetStructureSummary, suspicious: List[SuspiciousColumn]) -> DatasetQualityReport:
    """
    Calculate a transparent, deterministic dataset quality score (0 to 100).
    Penalties are clearly itemized so the user understands the exact deduction causes.
    """
    base_score = 100
    penalties = {
        "missing_data": 0,
        "duplicate_rows": 0,
        "constant_columns": 0,
        "high_cardinality": 0,
        "suspicious_ids": 0
    }

    # 1. Missing Data Penalty (Max: -30)
    avg_missing_pct = sum(c.missing_percentage for c in summary.columns) / max(summary.column_count, 1)
    missing_cols = [c.name for c in summary.columns if c.missing_count > 0]
    if avg_missing_pct > 0:
        # Scale: 0 to 30 penalty
        penalties["missing_data"] = min(30, int(avg_missing_pct * 0.75) + (5 if missing_cols else 0))

    # 2. Duplicate Rows Penalty (Max: -20)
    if summary.duplicate_rows_percentage > 0:
        penalties["duplicate_rows"] = min(20, int(summary.duplicate_rows_percentage * 1.5) + 3)

    # 3. Constant Columns Penalty (Max: -15)
    constant_cols = [c.name for c in summary.columns if c.is_constant]
    if constant_cols:
        penalties["constant_columns"] = min(15, len(constant_cols) * 5)

    # 4. High Cardinality Penalty (Max: -15)
    high_card_cols = [c.name for c in summary.columns if c.is_high_cardinality]
    if high_card_cols:
        penalties["high_cardinality"] = min(15, len(high_card_cols) * 4)

    # 5. Suspicious IDs Penalty (Max: -10)
    id_cols = [s.column_name for s in suspicious if s.issue_type == "id_like"]
    if id_cols:
        penalties["suspicious_ids"] = min(10, len(id_cols) * 3)

    total_penalty = sum(penalties.values())
    final_score = max(0, min(100, base_score - total_penalty))

    if final_score >= 85:
        rating = "Excellent"
    elif final_score >= 70:
        rating = "Good"
    elif final_score >= 50:
        rating = "Fair"
    else:
        rating = "Needs Attention"

    # Assemble Quality Cards / Checks
    checks = [
        QualityCheck(
            name="Missing Data",
            status="pass" if not missing_cols else ("warning" if avg_missing_pct < 20 else "fail"),
            summary=f"{len(missing_cols)} of {summary.column_count} columns contain missing values (avg {avg_missing_pct:.1f}%).",
            description="High missingness requires automated imputation and can degrade model predictive performance.",
            affected_columns=missing_cols
        ),
        QualityCheck(
            name="Duplicate Rows",
            status="pass" if summary.duplicate_rows_count == 0 else ("warning" if summary.duplicate_rows_percentage < 10 else "fail"),
            summary=f"{summary.duplicate_rows_count} duplicate rows found ({summary.duplicate_rows_percentage}%).",
            description="Duplicate rows can falsely inflate validation scores or introduce training bias.",
            affected_columns=[]
        ),
        QualityCheck(
            name="Constant Columns",
            status="pass" if not constant_cols else "warning",
            summary=f"{len(constant_cols)} constant columns detected.",
            description="Constant columns contain only 1 distinct value and provide zero variance.",
            affected_columns=constant_cols
        ),
        QualityCheck(
            name="High Cardinality",
            status="pass" if not high_card_cols else "warning",
            summary=f"{len(high_card_cols)} high-cardinality categorical features flagged.",
            description="Categorical columns with many distinct levels can increase model complexity or cause overfitting.",
            affected_columns=high_card_cols
        ),
        QualityCheck(
            name="Suspicious Identifiers",
            status="pass" if not id_cols else "warning",
            summary=f"{len(id_cols)} ID-like columns detected.",
            description="Features that behave as unique identifiers should be excluded from model training to prevent leakage.",
            affected_columns=id_cols
        )
    ]

    return DatasetQualityReport(
        quality_score=final_score,
        rating=rating,
        checks=checks,
        suspicious_columns=suspicious,
        score_breakdown=penalties
    )
