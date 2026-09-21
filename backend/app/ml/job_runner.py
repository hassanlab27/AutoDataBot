import time
import threading
import json
import logging
from typing import Dict, Optional, Any, Callable
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class RunStatus(BaseModel):
    run_id: str
    dataset_id: str
    status: str = "created"  # "created", "queued", "running", "evaluating", "completed", "failed", "cancelled"
    current_stage: str = "Initialized"
    current_engine: Optional[str] = None
    progress_pct: int = 0
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None
    cancellation_requested: bool = False

class JobRunner:
    _instance: Optional["JobRunner"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "JobRunner":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._runs: Dict[str, RunStatus] = {}
                cls._instance._threads: Dict[str, threading.Thread] = {}
                cls._instance._run_dirs: Dict[str, Path] = {}
            return cls._instance

    def register_run(self, run_id: str, dataset_id: str, run_dir: Path) -> RunStatus:
        with self._lock:
            status = RunStatus(run_id=run_id, dataset_id=dataset_id)
            self._runs[run_id] = status
            self._run_dirs[run_id] = run_dir
            self._save_status_to_disk(run_id)
            return status

    def get_status(self, run_id: str) -> Optional[RunStatus]:
        with self._lock:
            if run_id in self._runs:
                st = self._runs[run_id]
                if st.status in ["running", "evaluating", "queued"]:
                    st.elapsed_seconds = round(time.time() - st.start_time, 1)
                return st
        
        # Check disk if not in memory
        disk_status = self._load_status_from_disk(run_id)
        if disk_status:
            with self._lock:
                self._runs[run_id] = disk_status
            return disk_status
        return None

    def update_status(
        self,
        run_id: str,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        engine: Optional[str] = None,
        progress_pct: Optional[int] = None,
        error: Optional[str] = None
    ) -> None:
        with self._lock:
            if run_id not in self._runs:
                return
            st = self._runs[run_id]
            if status:
                st.status = status
            if stage:
                st.current_stage = stage
            if engine is not None:
                st.current_engine = engine
            if progress_pct is not None:
                st.progress_pct = max(0, min(100, progress_pct))
            if error:
                st.error_message = error

            st.elapsed_seconds = round(time.time() - st.start_time, 1)
            if st.status in ["completed", "failed", "cancelled"]:
                st.end_time = time.time()
                st.elapsed_seconds = round(st.end_time - st.start_time, 1)

            self._save_status_to_disk(run_id)

    def request_cancellation(self, run_id: str) -> bool:
        with self._lock:
            if run_id in self._runs:
                st = self._runs[run_id]
                if st.status in ["completed", "failed", "cancelled"]:
                    return False
                st.cancellation_requested = True
                st.status = "cancelled"
                st.current_stage = "Cancellation requested by user"
                st.end_time = time.time()
                st.elapsed_seconds = round(st.end_time - st.start_time, 1)
                self._save_status_to_disk(run_id)
                return True
        return False

    def is_cancelled(self, run_id: str) -> bool:
        with self._lock:
            if run_id in self._runs:
                return self._runs[run_id].cancellation_requested
        return False

    def start_job(self, run_id: str, target_fn: Callable[..., Any], args: tuple = ()) -> None:
        def worker():
            try:
                self.update_status(run_id, status="running", stage="Training started", progress_pct=5)
                target_fn(*args)
            except Exception as e:
                logger.exception(f"Unhandled error in AutoML run {run_id}: {e}")
                self.update_status(run_id, status="failed", stage="Execution failed", error=str(e))

        thread = threading.Thread(target=worker, daemon=True, name=f"automl-worker-{run_id}")
        with self._lock:
            self._threads[run_id] = thread
        thread.start()

    def _save_status_to_disk(self, run_id: str) -> None:
        run_dir = self._run_dirs.get(run_id)
        if not run_dir or not run_dir.exists():
            return
        status_file = run_dir / "status.json"
        try:
            with open(status_file, "w", encoding="utf-8") as f:
                f.write(self._runs[run_id].model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"Could not persist status for run {run_id}: {e}")

    def _load_status_from_disk(self, run_id: str) -> Optional[RunStatus]:
        from app.core.config import settings
        runs_dir = settings.OUTPUTS_DIR / "runs" / run_id
        status_file = runs_dir / "status.json"
        if status_file.exists():
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return RunStatus(**data)
            except Exception as e:
                logger.warning(f"Could not load status for run {run_id} from disk: {e}")
        return None

job_runner = JobRunner()
