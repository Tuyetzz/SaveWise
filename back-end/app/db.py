"""SQLite persistence via SQLModel. No migrations: on schema change, delete
data/triage.db and let create_all recreate it."""

from datetime import datetime

from sqlmodel import JSON, Column, Field, Session, SQLModel, create_engine


class Interview(SQLModel, table=True):
    id: str = Field(primary_key=True)
    status: str = "active"  # active | complete | abandoned | no_response
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: datetime | None = None


class Turn(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    interview_id: str = Field(index=True)
    seq: int
    question_id: str | None = None
    question_text: str | None = None
    transcript: str | None = None
    audio_path: str | None = None
    stt_ms: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Assessment(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    interview_id: str = Field(index=True)
    turn_seq: int
    fields: dict = Field(sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CaseEvent(SQLModel, table=True):
    """Append-only responder actions on a case: category overrides, workflow
    changes (dispatched/rescued), and free-text notes. Latest event of a kind
    wins; the full log is the audit trail."""

    id: int | None = Field(default=None, primary_key=True)
    interview_id: str = Field(index=True)
    kind: str  # override | workflow | note
    value: str | None = None
    note: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


engine = create_engine(
    "sqlite:///data/triage.db",
    connect_args={"check_same_thread": False},
)


def create_all() -> None:
    SQLModel.metadata.create_all(engine)


__all__ = [
    "Interview",
    "Turn",
    "Assessment",
    "CaseEvent",
    "engine",
    "Session",
    "create_all",
]
