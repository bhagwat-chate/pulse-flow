import uuid
import contextvars


# Context variable to hold trace_id per request
_trace_id_ctx = contextvars.ContextVar("trace_id", default=None)


def new_trace_id() -> str:
    """Generate a new unique trace_id and set it in context."""
    trace_id = str(uuid.uuid4())
    _trace_id_ctx.set(trace_id)
    return trace_id


def get_trace_id() -> str:
    """Return the current trace_id from context, if any."""
    return _trace_id_ctx.get() or "no-trace-id"
