from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

# Create database engine
# We use connect_args={"check_same_thread": False} for SQLite compatibility with multiple threads
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_async_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

# Create async session maker
SessionLocal = async_sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

# Declarative base class for models
class Base(DeclarativeBase):
    pass

# DB session dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Helper to initialize DB tables
async def init_db() -> None:
    async with engine.begin() as conn:
        # Import models here to register them with Base metadata
        from app.db.models import Challan, VehicleLog
        await conn.run_sync(Base.metadata.create_all)
