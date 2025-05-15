import dataclasses
import json

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType
from pipecat.services.ojin.ojin_avatar_messages import OjinAvatarMessage


class OjinAvatarSerializer(FrameSerializer):
    def __init__(self):
        super().__init__(FrameSerializerType.JSON)

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize a frame to JSON string if it is a TransportMessageFrame or TransportMessageUrgentFrame.

        Args:
            frame (Frame): The frame to serialize.

        Returns:
            str | bytes | None: The JSON string representation of the frame, or None if the frame is not a supported type.

        """
        if isinstance(frame, TransportMessageFrame | TransportMessageUrgentFrame):
            message = frame.message
            if isinstance(message, OjinAvatarMessage):
                return json.dumps(dataclasses.asdict(message))
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        try:
            message: OjinAvatarMessage = json.loads(data)
            return TransportMessageFrame(message=message)
        except Exception as e:
            logger.error(f"Exception deserializing OjinAvatarMessage: {e}")
            return None
