"""消息系统"""

from datetime import datetime
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel


MessagesRole = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    '''消息类'''
    content:str
    role:MessagesRole
    timestamp:datetime = None
    metadate:Optional[dict[str, Any]] = None
    def __init__(self, content: str, role: MessagesRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get('timestamp', datetime.now()),
            metadata=kwargs.get('metadata', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（OpenAI API格式）"""
        return {
            "role": self.role,
            "content": self.content
        }
    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
