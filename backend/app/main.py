from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.errors import AutoDataBotException
from app.core.logging import logger
from app.api.routes.health import router as health_router
from app.api.routes.datasets import router as datasets_router
from app.api.routes.eda import router as eda_router
from app.api.routes.preprocessing import router as preprocessing_router
from app.api.routes.runs import router as runs_router
from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.explainability import router as explainability_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}...")
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Exception Handler for Domain Errors
@app.exception_handler(AutoDataBotException)
async def domain_exception_handler(request: Request, exc: AutoDataBotException):
    logger.warning(f"Domain error [{exc.error}]: {exc.message} (Path: {request.url.path})")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error,
            "message": exc.message,
            "details": exc.details
        }
    )

# Generic Exception Handler (hide internal stack traces)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected server error occurred while processing the request.",
            "details": {}
        }
    )

# Register API Routers
app.include_router(health_router, prefix=settings.API_PREFIX)
app.include_router(datasets_router, prefix=settings.API_PREFIX)
app.include_router(eda_router, prefix=settings.API_PREFIX)
app.include_router(preprocessing_router, prefix=settings.API_PREFIX)
app.include_router(runs_router, prefix=settings.API_PREFIX)
app.include_router(evaluation_router, prefix=settings.API_PREFIX)
app.include_router(explainability_router, prefix=settings.API_PREFIX)

@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/health"
    }
