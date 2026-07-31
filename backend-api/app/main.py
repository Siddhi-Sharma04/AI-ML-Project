import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.session import init_db
from app.api.endpoints.challans import router as challan_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB tables
    await init_db()
    
    # Ensure media directory exists
    os.makedirs(settings.MEDIA_DIR, exist_ok=True)
    
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS middleware setup
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include endpoint routes
app.include_router(
    challan_router,
    prefix=f"{settings.API_V1_STR}/challans",
    tags=["challans"]
)

# Mount media directory to serve evidence files (images/PDFs) static
app.mount("/media", StaticFiles(directory=settings.MEDIA_DIR), name="media")

@app.get("/")
def read_root():
    return {
        "status": "ONLINE",
        "project": settings.PROJECT_NAME,
        "api_v1_docs": "/docs"
    }
