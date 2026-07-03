from __future__ import annotations
import uuid
from ..core.types import Message


class ConversationStore:
    """
    In-memory session store.

    Each session is a list of user/assistant messages (no system messages).
    History is trimmed to max_messages to avoid exceeding Claude's context window.
    State is lost on server restart — swap the dict for Redis/DB for persistence.
    """

    def __init__(self, max_messages: int = 50):
        self._sessions: dict[str, list[Message]] = {}
        self._max_messages = max_messages

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = []
        return session_id

    def get(self, session_id: str) -> list[Message]:
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, user_msg: Message, assistant_msg: Message) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].extend([user_msg, assistant_msg])
        # Keep only the most recent messages to stay within context limits
        if len(self._sessions[session_id]) > self._max_messages:
            self._sessions[session_id] = self._sessions[session_id][-self._max_messages:]

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def message_count(self, session_id: str) -> int:
        return len(self._sessions.get(session_id, []))
