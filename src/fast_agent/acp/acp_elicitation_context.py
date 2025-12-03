"""
State management for ACP elicitation Q&A sessions.

This module tracks the state of an interactive elicitation session,
including which question we're on, collected answers, and validation state.
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import AnyUrl, BaseModel, EmailStr
from pydantic import ValidationError as PydanticValidationError

from fast_agent.acp.elicitation_questions import (
    ElicitationQuestion,
    format_completion_summary,
    format_elicitation_intro,
    format_question_for_display,
    schema_to_questions,
)
from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


class ElicitationPhase(Enum):
    """Current phase of the elicitation session."""

    INTRO = "intro"  # Showing introduction, waiting for user to proceed
    QUESTIONING = "questioning"  # Asking questions one by one
    CONFIRMATION = "confirmation"  # Showing summary, waiting for confirmation
    COMPLETED = "completed"  # Session finished
    CANCELLED = "cancelled"  # User cancelled
    DECLINED = "declined"  # User declined to provide info


@dataclass
class ElicitationResult:
    """Result of an elicitation session."""

    action: str  # "accept", "decline", "cancel"
    data: dict[str, Any] | None = None


@dataclass
class ACPElicitationContext:
    """
    Manages the state of an interactive elicitation session over ACP.

    This context tracks:
    - The schema being elicited
    - Which question we're currently on
    - Answers collected so far
    - The phase of the session (intro, questioning, confirmation)
    """

    session_id: str
    """ACP session ID this elicitation belongs to."""

    schema: dict[str, Any]
    """The JSON Schema being elicited."""

    message: str
    """The elicitation message from the MCP server."""

    agent_name: str
    """Name of the agent handling this elicitation."""

    server_name: str
    """Name of the MCP server requesting elicitation."""

    # Parsed questions
    questions: list[ElicitationQuestion] = field(default_factory=list)

    # Current state
    phase: ElicitationPhase = ElicitationPhase.INTRO
    current_question_index: int = 0
    collected_answers: dict[str, Any] = field(default_factory=dict)

    # For async waiting
    _response_event: asyncio.Event = field(default_factory=asyncio.Event)
    _pending_response: str | None = None
    _result: ElicitationResult | None = None

    def __post_init__(self):
        """Parse the schema into questions after initialization."""
        self.questions = schema_to_questions(self.schema)

    @property
    def total_questions(self) -> int:
        """Total number of questions."""
        return len(self.questions)

    @property
    def current_question(self) -> ElicitationQuestion | None:
        """Get the current question, or None if done."""
        if 0 <= self.current_question_index < len(self.questions):
            return self.questions[self.current_question_index]
        return None

    @property
    def is_complete(self) -> bool:
        """Check if the elicitation is complete."""
        return self.phase in (
            ElicitationPhase.COMPLETED,
            ElicitationPhase.CANCELLED,
            ElicitationPhase.DECLINED,
        )

    def get_intro_message(self) -> str:
        """Get the introduction message to display."""
        return format_elicitation_intro(
            schema=self.schema,
            message=self.message,
            agent_name=self.agent_name,
            server_name=self.server_name,
            total_questions=self.total_questions,
        )

    def get_current_question_message(self) -> str | None:
        """Get the formatted current question message."""
        question = self.current_question
        if question is None:
            return None

        return format_question_for_display(
            question=question,
            question_number=self.current_question_index + 1,
            total_questions=self.total_questions,
        )

    def get_confirmation_message(self) -> str:
        """Get the confirmation summary message."""
        return format_completion_summary(
            schema=self.schema,
            collected_data=self.collected_answers,
        )

    def process_intro_response(self, response: str) -> str | None:
        """
        Process user response during intro phase.

        Returns the next message to display, or None if moving to questions.
        """
        response_lower = response.strip().lower()

        if response_lower in ("cancel", "quit", "exit", "q"):
            self.phase = ElicitationPhase.CANCELLED
            self._result = ElicitationResult(action="cancel")
            return None

        if response_lower in ("decline", "no", "reject"):
            self.phase = ElicitationPhase.DECLINED
            self._result = ElicitationResult(action="decline")
            return None

        # Any other response (including empty) moves to questioning
        self.phase = ElicitationPhase.QUESTIONING
        return self.get_current_question_message()

    def process_question_response(self, response: str) -> tuple[str | None, str | None]:
        """
        Process user response to a question.

        Returns:
            (next_message, error_message) - error_message is set if validation failed
        """
        response_stripped = response.strip()
        response_lower = response_stripped.lower()

        # Handle special commands
        if response_lower in ("cancel", "quit", "exit", "q"):
            self.phase = ElicitationPhase.CANCELLED
            self._result = ElicitationResult(action="cancel")
            return None, None

        if response_lower in ("decline",):
            self.phase = ElicitationPhase.DECLINED
            self._result = ElicitationResult(action="decline")
            return None, None

        if response_lower == "back":
            if self.current_question_index > 0:
                self.current_question_index -= 1
                # Remove previous answer if it exists
                prev_question = self.current_question
                if prev_question and prev_question.field_name in self.collected_answers:
                    del self.collected_answers[prev_question.field_name]
            return self.get_current_question_message(), None

        if response_lower == "skip":
            question = self.current_question
            if question and not question.required:
                # Use default if available, otherwise None
                self.collected_answers[question.field_name] = question.default
                return self._advance_to_next_question(), None
            else:
                return None, "This field is required and cannot be skipped."

        # Validate and store the answer
        question = self.current_question
        if question is None:
            return None, "No current question to answer."

        # Handle empty response
        if not response_stripped:
            if question.default is not None:
                # Use default value
                self.collected_answers[question.field_name] = question.default
                return self._advance_to_next_question(), None
            elif question.required:
                return None, "This field is required. Please provide a value."
            else:
                # Optional field with no default - skip it
                self.collected_answers[question.field_name] = None
                return self._advance_to_next_question(), None

        # Validate the response
        validated_value, error = self._validate_answer(question, response_stripped)
        if error:
            return None, error

        # Store the validated answer
        self.collected_answers[question.field_name] = validated_value
        return self._advance_to_next_question(), None

    def process_confirmation_response(self, response: str) -> str | None:
        """
        Process user response during confirmation phase.

        Returns the next message to display, or None if complete.
        """
        response_lower = response.strip().lower()

        if response_lower in ("submit", "confirm", "yes", "y", "ok"):
            self.phase = ElicitationPhase.COMPLETED
            self._result = ElicitationResult(action="accept", data=self.collected_answers)
            return None

        if response_lower in ("cancel", "quit", "exit", "q"):
            self.phase = ElicitationPhase.CANCELLED
            self._result = ElicitationResult(action="cancel")
            return None

        if response_lower in ("decline",):
            self.phase = ElicitationPhase.DECLINED
            self._result = ElicitationResult(action="decline")
            return None

        if response_lower == "back":
            # Go back to the last question
            if self.questions:
                self.current_question_index = len(self.questions) - 1
                self.phase = ElicitationPhase.QUESTIONING
                # Remove the last answer so user can re-enter
                last_question = self.current_question
                if last_question and last_question.field_name in self.collected_answers:
                    del self.collected_answers[last_question.field_name]
                return self.get_current_question_message()
            return self.get_confirmation_message()

        # Unrecognized - show confirmation again with hint
        return (
            "Please type `submit` to confirm, `back` to edit, or `cancel` to abort.\n\n"
            + self.get_confirmation_message()
        )

    def _advance_to_next_question(self) -> str | None:
        """
        Move to the next question or confirmation phase.

        Returns the next message to display.
        """
        self.current_question_index += 1

        if self.current_question_index >= len(self.questions):
            # All questions answered - move to confirmation
            self.phase = ElicitationPhase.CONFIRMATION
            return self.get_confirmation_message()

        return self.get_current_question_message()

    def _validate_answer(
        self, question: ElicitationQuestion, answer: str
    ) -> tuple[Any, str | None]:
        """
        Validate an answer against the question's constraints.

        Returns:
            (validated_value, error_message) - error_message is None if valid
        """
        try:
            if question.field_type == "boolean":
                return self._validate_boolean(answer)

            elif question.field_type == "enum":
                return self._validate_enum(question, answer)

            elif question.field_type == "integer":
                return self._validate_integer(question, answer)

            elif question.field_type == "number":
                return self._validate_number(question, answer)

            else:  # string
                return self._validate_string(question, answer)

        except Exception as e:
            logger.warning(f"Validation error: {e}")
            return None, str(e)

    def _validate_boolean(self, answer: str) -> tuple[bool | None, str | None]:
        """Validate a boolean answer."""
        answer_lower = answer.lower()
        if answer_lower in ("yes", "y", "true", "1", "on"):
            return True, None
        elif answer_lower in ("no", "n", "false", "0", "off"):
            return False, None
        else:
            return None, "Please answer yes/no (or y/n, true/false)."

    def _validate_enum(
        self, question: ElicitationQuestion, answer: str
    ) -> tuple[str | None, str | None]:
        """Validate an enum answer."""
        # Try matching by number
        try:
            num = int(answer)
            if 1 <= num <= len(question.options):
                return question.options[num - 1][0], None
        except ValueError:
            pass

        # Try matching by value
        answer_lower = answer.lower()
        for value, label in question.options:
            if answer_lower == value.lower() or answer_lower == label.lower():
                return value, None

        options_display = ", ".join([f"{i+1}={label}" for i, (_, label) in enumerate(question.options)])
        return None, f"Please choose one of: {options_display}"

    def _validate_integer(
        self, question: ElicitationQuestion, answer: str
    ) -> tuple[int | None, str | None]:
        """Validate an integer answer."""
        try:
            value = int(answer)
        except ValueError:
            return None, "Please enter a valid whole number."

        if question.min_value is not None and value < question.min_value:
            return None, f"Value must be at least {question.min_value}."
        if question.max_value is not None and value > question.max_value:
            return None, f"Value must be at most {question.max_value}."

        return value, None

    def _validate_number(
        self, question: ElicitationQuestion, answer: str
    ) -> tuple[float | None, str | None]:
        """Validate a number answer."""
        try:
            value = float(answer)
        except ValueError:
            return None, "Please enter a valid number."

        if question.min_value is not None and value < question.min_value:
            return None, f"Value must be at least {question.min_value}."
        if question.max_value is not None and value > question.max_value:
            return None, f"Value must be at most {question.max_value}."

        return value, None

    def _validate_string(
        self, question: ElicitationQuestion, answer: str
    ) -> tuple[str | None, str | None]:
        """Validate a string answer."""
        # Length constraints
        if question.min_length is not None and len(answer) < question.min_length:
            return None, f"Please enter at least {question.min_length} characters."
        if question.max_length is not None and len(answer) > question.max_length:
            return None, f"Please enter at most {question.max_length} characters."

        # Pattern constraint
        if question.pattern:
            if not re.fullmatch(question.pattern, answer):
                return None, f"Value must match pattern: {question.pattern}"

        # Format constraints
        if question.format_type:
            error = self._validate_format(question.format_type, answer)
            if error:
                return None, error

        return answer, None

    def _validate_format(self, format_type: str, value: str) -> str | None:
        """Validate a format constraint. Returns error message or None."""
        try:
            if format_type == "email":
                class EmailModel(BaseModel):
                    email: EmailStr
                EmailModel(email=value)

            elif format_type == "uri":
                class UriModel(BaseModel):
                    uri: AnyUrl
                UriModel(uri=value)

            elif format_type == "date":
                date.fromisoformat(value)

            elif format_type == "date-time":
                datetime.fromisoformat(value.replace("Z", "+00:00"))

            return None  # Valid

        except (PydanticValidationError, ValueError):
            format_hints = {
                "email": "Please enter a valid email address (e.g., user@example.com).",
                "uri": "Please enter a valid URL (e.g., https://example.com).",
                "date": "Please enter a valid date in YYYY-MM-DD format.",
                "date-time": "Please enter a valid date-time in ISO 8601 format.",
            }
            return format_hints.get(format_type, f"Invalid {format_type} format.")

    def get_result(self) -> ElicitationResult | None:
        """Get the final result of the elicitation, or None if not complete."""
        return self._result

    # Methods for async waiting (used when integrating with ACP)
    def submit_response(self, response: str) -> None:
        """Submit a user response (called by ACP server when prompt received)."""
        self._pending_response = response
        self._response_event.set()

    async def wait_for_response(self) -> str:
        """Wait for a user response (called by elicitation handler)."""
        await self._response_event.wait()
        self._response_event.clear()
        response = self._pending_response or ""
        self._pending_response = None
        return response
