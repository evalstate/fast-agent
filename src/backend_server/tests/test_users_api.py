import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend_server.main import app
from database.database import Base, get_db
from database.models import User

# Get test database URL from environment variable or use a default
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://admin:admin@34.47.187.64:5432/postgres")
engine = create_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override the dependency with our test database


def test_create_user(client):
    """Test creating a new user"""
    # Test data
    user_data = {
        "address": "0x123456789abcdef",
        "points": 100,
        "position": 1
    }
    
    # Send POST request to create user
    response = client.post("/users/", json=user_data)
    
    # Check response
    assert response.status_code == 201
    data = response.json()
    assert data["address"] == user_data["address"]
    assert data["points"] == user_data["points"]
    assert data["position"] == user_data["position"]


def test_get_users(client):
    """Test retrieving all users"""
    # Create test users first
    
    
    # Test GET users endpoint
    response = client.get("/users/")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    print(data)