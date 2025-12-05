"""
Database factory to handle both SQLite and PostgreSQL implementations.
Allows gradual migration from SQLite to PostgreSQL with feature flag control.
"""

import os
from typing import Union
from .database import Database, get_database
from .database_postgres import PostgreSQLDatabase, get_postgres_database


class DatabaseFactory:
    """Factory class to create appropriate database instance based on configuration."""
    
    @staticmethod
    def create_database() -> Union[Database, PostgreSQLDatabase]:
        """Create database instance based on USE_POSTGRESQL environment variable."""
        use_postgresql = os.getenv("USE_POSTGRESQL", "false").lower() == "true"
        
        if use_postgresql:
            return get_postgres_database()
        else:
            return get_database()
    
    @staticmethod
    async def create_database_async() -> Union[Database, PostgreSQLDatabase]:
        """Create and initialize database instance based on USE_POSTGRESQL environment variable."""
        db = DatabaseFactory.create_database()
        await db.initialize()
        return db
    
    @staticmethod
    def get_database_type() -> str:
        """Get the current database type being used."""
        use_postgresql = os.getenv("USE_POSTGRESQL", "false").lower() == "true"
        return "PostgreSQL" if use_postgresql else "SQLite"


# Global factory instance
_factory_instance = None

def get_database_factory() -> DatabaseFactory:
    """Get the global database factory instance."""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = DatabaseFactory()
    return _factory_instance


# Convenience functions that work with both database types
def get_current_database() -> Union[Database, PostgreSQLDatabase]:
    """Get the currently configured database instance."""
    return DatabaseFactory.create_database()


async def initialize_database():
    """Initialize the currently configured database."""
    db = get_current_database()
    await db.initialize()
    return db


async def close_database():
    """Close the currently configured database."""
    db = get_current_database()
    if hasattr(db, 'close'):  # PostgreSQL has close method
        await db.close()
    # SQLite doesn't need explicit close since it uses connection contexts