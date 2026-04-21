"""Typed errors for session trace export."""

from __future__ import annotations


class TraceExportError(Exception):
    """Base error for session trace export failures."""


class InvalidSessionExportTargetError(TraceExportError):
    """Raised when a session export target cannot be resolved."""


class SessionExportNotFoundError(TraceExportError):
    """Raised when the requested session cannot be found."""


class SessionExportNoAgentsError(TraceExportError):
    """Raised when a session has no exportable agents."""


class SessionExportAgentNotFoundError(TraceExportError):
    """Raised when a requested agent is not exportable for a session."""


class SessionExportAmbiguousAgentError(TraceExportError):
    """Raised when a session has multiple exportable agents and none was specified."""


class UnsupportedTraceExportFormatError(TraceExportError):
    """Raised when a caller requests an unknown export format."""


class SessionExportUploadError(TraceExportError):
    """Raised when an exported trace cannot be uploaded to a remote destination."""


class SessionExportReadError(TraceExportError):
    """Raised when a persisted session cannot be read for export."""


class SessionExportWriteError(TraceExportError):
    """Raised when an exported trace cannot be written to disk."""
