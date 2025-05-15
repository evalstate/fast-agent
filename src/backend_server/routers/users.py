from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from src.database.models import User
from src.database.database import get_db
from pydantic import BaseModel

router = APIRouter(
    prefix="/users",
    tags=["users"],
)


class UserCreate(BaseModel):
    address: str
    points: int = 0
    position: int = None


class UserResponse(BaseModel):
    address: str
    points: int
    position: int = None

    class Config:
        orm_mode = True


@router.get("/", response_model=List[UserResponse])
async def get_users(db: Session = Depends(get_db)):
    """
    Retrieve all users.
    """
    users = db.query(User).all()
    return users


@router.post("/", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Create a new user.
    """
    # Check if user already exists
    db_user = db.query(User).filter(User.address == user.address).first()
    if db_user:
        raise HTTPException(status_code=400, detail="User with this address already exists")
    
    # Create new user
    new_user = User(
        address=user.address,
        points=user.points,
        position=user.position
    )
    
    # Add to database
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user