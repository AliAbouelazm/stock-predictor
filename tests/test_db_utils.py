"""Tests for database utilities."""

import pytest
import tempfile
from pathlib import Path
import pandas as pd

from src.database.db_utils import get_connection, initialize_schema, get_or_create_symbol, insert_prices


def test_database_initialization():
    """Test database schema initialization."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        conn = get_connection(db_path)
        initialize_schema(conn)
        
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "symbols" in tables
        assert "prices" in tables
        assert "features" in tables
        assert "targets" in tables
        assert "predictions" in tables
        
        conn.close()
    finally:
        if db_path.exists():
            db_path.unlink()


def test_symbol_creation():
    """Test symbol creation and retrieval."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        conn = get_connection(db_path)
        initialize_schema(conn)
        
        symbol_id = get_or_create_symbol(conn, "TEST", "Test Company")
        assert symbol_id is not None
        
        symbol_id2 = get_or_create_symbol(conn, "TEST", "Test Company")
        assert symbol_id == symbol_id2
        
        conn.close()
    finally:
        if db_path.exists():
            db_path.unlink()

