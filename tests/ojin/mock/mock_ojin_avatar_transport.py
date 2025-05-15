import asyncio
import math
import uuid

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ojin.ojin_avatar_messages import (
    AuthenticationMessage,
    AuthenticationResponseMessage,
    InteractionInputMessage,
    InteractionReadyMessage,
    InteractionResponseMessage,
    IOjinAvatarClient,
    OjinAvatarInputFrame,
    OjinAvatarMessage,
    StartInteractionMessage,
)


class MockOjinAvatarClient(IOjinAvatarClient):
    def __init__(self):
        self._auth_token: str | None = None
        self._interaction_id: str | None = None
        self._interaction_input: str | None = None
        self._output_queue: asyncio.Queue[OjinAvatarMessage] = asyncio.Queue()
        self._avatar_id: str | None = None
        self._video_frames_task: asyncio.Task | None = asyncio.create_task(
            self._send_video_frames()
        )
        self._num_pending_video_frames = 0
        self._frame_idx = 0

        # Message counters
        self.messages_received: dict[type[OjinAvatarMessage], int] = {}
        self.messages_received.setdefault(AuthenticationResponseMessage, 0)
        self.messages_received.setdefault(InteractionReadyMessage, 0)
        self.messages_received.setdefault(InteractionResponseMessage, 0)

    def close(self):
        self._video_frames_task.cancel()

    async def _send_video_frames(self):
        while True:
            await asyncio.sleep(0.03)
            if self._interaction_id is None:
                continue

            if self._num_pending_video_frames <= 0:
                self._frame_idx = 0
                continue

            response = InteractionResponseMessage(
                interaction_id=self._interaction_id,
                avatar_id=self._avatar_id,
                payload=f"frame_{self._frame_idx}".encode(),
                frame_idx=self._frame_idx,
                is_final_response=self._num_pending_video_frames == 1,
            )
            self._num_pending_video_frames -= 1
            self._frame_idx += 1
            self._output_queue.put_nowait(response)

    def send_message(self, message: OjinAvatarMessage) -> None:
        if isinstance(message, AuthenticationMessage):
            logger.info("AuthenticationMessage")
            assert message.api_key is not None
            self._output_queue.put_nowait(
                AuthenticationResponseMessage(token="mock_token")
            )

        elif isinstance(message, StartInteractionMessage):
            logger.info("StartInteractionMessage")
            self._interaction_id = str(uuid.uuid4())
            self._avatar_id = message.avatar_id
            self._num_pending_video_frames = 0
            self._frame_idx = 0
            self._output_queue.put_nowait(
                InteractionReadyMessage(
                    interaction_id=self._interaction_id, avatar_id=self._avatar_id
                )
            )

        elif isinstance(message, InteractionInputMessage):
            logger.info(
                f"InteractionInputMessage is_last_input: {message.is_last_input} interrupt: {message.interrupt}"
            )
            if message.interrupt:
                self._num_pending_video_frames = 0
                self._frame_idx = 0
                while not self._output_queue.empty():
                    self._output_queue.get_nowait()

            elif message.audio is not None:
                self._num_pending_video_frames += self.get_expected_frames(
                    message.audio
                )
                logger.info(
                    f"Adding {self.get_expected_frames(message.audio)} expected frames, total: {self._num_pending_video_frames}"
                )

    def get_expected_frames(self, audio: bytes) -> int:
        """Calculate expected frames based on the wav file duration."""
        expected_frames = math.ceil(len(audio) / 640)
        return expected_frames

    async def receive_message(self) -> OjinAvatarMessage:
        message = await self._output_queue.get()
        self.messages_received[type(message)] += 1
        return message


class MockOjinAvatarTransport:
    def __init__(self, client: IOjinAvatarClient):
        self._input = MockOjinAvatarTransportInput(client)
        self._output = MockOjinAvatarTransportOutput(client)

    def input(self):
        return self._input

    def output(self):
        return self._output


class MockOjinAvatarTransportInput(FrameProcessor):
    def __init__(self, client: IOjinAvatarClient):
        super().__init__()
        self._client = client

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._start()
        elif isinstance(frame, EndFrame | CancelFrame):
            await self._end()

        await self.push_frame(frame, direction)

    def _start(self):
        self._receiver_task = self.create_task(self._receive_messages())

    async def _end(self):
        if self._receiver_task:
            await self.cancel_task(self._receiver_task)
            self._receiver_task = None

        self._client.close()

    async def _receive_messages(self):
        while True:
            message = await self._client.receive_message()
            await self.push_frame(OjinAvatarInputFrame(message=message))


class MockOjinAvatarTransportOutput(FrameProcessor):
    def __init__(self, client: IOjinAvatarClient):
        super().__init__()
        self._client = client

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TransportMessageFrame | TransportMessageUrgentFrame):
            self._client.send_message(frame.message)

        await self.push_frame(frame, direction)
