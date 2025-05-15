# database/models.py

from sqlalchemy import (
    Column, Integer, String, DateTime,
    ForeignKey, Index, Enum
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from .database import Base


class MCP(Base):
    __tablename__ = "mcps"

    id               = Column(Integer, primary_key=True, index=True)
    mcp_name         = Column(String, index=True)
    user_address     = Column(
                         String,
                         ForeignKey("users.address", ondelete="CASCADE"),
                         index=True
                       )
    mcp_json         = Column(JSONB)
    mcp_env_keys     = Column(JSONB)
    tool_calls_count = Column(Integer, default=0)

    # ORM relationship
    user = relationship("User", back_populates="mcps")


class User(Base):
    __tablename__ = "users"

    address      = Column(String, primary_key=True, index=True)
    points       = Column(Integer, default=0)
    position     = Column(Integer)

    # backrefs
    mcps          = relationship("MCP",         back_populates="user")
    queen_agents  = relationship("QueenAgent",  back_populates="user")
    worker_agents = relationship("WorkerAgent", back_populates="user")


class QueenAgent(Base):
    __tablename__ = "queen_agents"

    id               = Column(Integer, primary_key=True, index=True)
    user_address     = Column(
                         String,
                         ForeignKey("users.address", ondelete="CASCADE"),
                         index=True
                       )
    agent_slug       = Column(String, index=True)
    agent_name       = Column(String)
    agent_prompt     = Column(String)
    ipfs_hash        = Column(String)
    last_active      = Column(DateTime, default=func.now())
    number_of_calls  = Column(Integer, default=0)
    number_of_messages = Column(Integer, default=0)
    subagents_ids    = Column(JSONB, default=list)
    logs             = Column(JSONB, default=list)

    user = relationship("User", back_populates="queen_agents")


class WorkerAgent(Base):
    __tablename__ = "worker_agents"

    id             = Column(Integer, primary_key=True, index=True)
    user_address   = Column(
                       String,
                       ForeignKey("users.address", ondelete="CASCADE"),
                       index=True
                     )
    subagent_slug  = Column(String, index=True)
    subagent_name  = Column(String)
    mcp_connected  = Column(JSONB, default=list)
    logs           = Column(JSONB, default=list)
    model          = Column(String, default="o4-mini")
    total_calls    = Column(Integer, default=0)

    user = relationship("User", back_populates="worker_agents")


ActorType = Enum("user", "queen_agent", "worker_agent", name="actor_type")

class Message(Base):
    __tablename__ = "messages"
    id        = Column(Integer, primary_key=True, index=True)
    message   = Column(String, nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    cost      = Column(Integer, default=0, nullable=False)
    sender_type             = Column(ActorType, nullable=False)
    sender_user_address     = Column(
                                String,
                                ForeignKey("users.address", ondelete="CASCADE"),
                                nullable=True,
                            )
    sender_queen_agent_id   = Column(
                                Integer,
                                ForeignKey("queen_agents.id", ondelete="CASCADE"),
                                nullable=True,
                            )
    sender_worker_agent_id  = Column(
                                Integer,
                                ForeignKey("worker_agents.id", ondelete="CASCADE"),
                                nullable=True,
                            )

    receiver_type           = Column(ActorType, nullable=False)
    receiver_user_address   = Column(
                                String,
                                ForeignKey("users.address", ondelete="CASCADE"),
                                nullable=True,
                            )
    receiver_queen_agent_id = Column(
                                Integer,
                                ForeignKey("queen_agents.id", ondelete="CASCADE"),
                                nullable=True,
                            )
    receiver_worker_agent_id= Column(
                                Integer,
                                ForeignKey("worker_agents.id", ondelete="CASCADE"),
                                nullable=True,
                            )

    # handy ORM relationships
    sender_user        = relationship("User",        foreign_keys=[sender_user_address])
    sender_queen_agent = relationship("QueenAgent",  foreign_keys=[sender_queen_agent_id])
    sender_worker_agent= relationship("WorkerAgent", foreign_keys=[sender_worker_agent_id])

    receiver_user        = relationship("User",        foreign_keys=[receiver_user_address])
    receiver_queen_agent = relationship("QueenAgent",  foreign_keys=[receiver_queen_agent_id])
    receiver_worker_agent= relationship("WorkerAgent", foreign_keys=[receiver_worker_agent_id])


class Point(Base):
    __tablename__ = "points"

    id           = Column(Integer, primary_key=True, index=True)
    user_address = Column(
                     String,
                     ForeignKey("users.address", ondelete="CASCADE"),
                     index=True
                   )
    activity     = Column(String)
    points       = Column(Integer, default=0)

    user = relationship("User")
