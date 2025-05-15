"""Ojin Avatar implementation for Pipecat."""

import asyncio
from dataclasses import dataclass
from typing import Optional, Tuple

# Will use numpy when implementing avatar-specific processing
from loguru import logger

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputImageRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .ojin_avatar_messages import (
    AuthenticationMessage,
    AuthenticationResponseMessage,
    InteractionInputMessage,
    InteractionReadyMessage,
    InteractionResponseMessage,
    OjinAvatarInputFrame,
    OjinAvatarMessage,
    StartInteractionMessage,
)


@dataclass
class OjinAvatarSettings:
    """Settings for Ojin Avatar service.

    This class encapsulates all configuration parameters for the OjinAvatarService.
    """

    api_key: str | None = None
    ojin_avatar_id: str | None = None
    sample_rate: int = 16000
    image_size: Tuple[int, int] = (1920, 1080)


@dataclass
class OjinAvatarInteraction:
    interaction_id: str | None = None
    avatar_id: str | None = None
    audio_input_queue: asyncio.Queue = None
    is_running: bool = False
    is_ending: bool = False

    def __post_init__(self):
        """Initialize queues after instance creation."""
        if self.audio_input_queue is None:
            self.audio_input_queue = asyncio.Queue()

    def close(self):
        """Close the interaction."""
        while not self.audio_input_queue.empty():
            if self.audio_input_queue.empty():
                break
            self.audio_input_queue.get_nowait()
            self.audio_input_queue.task_done()

        self.is_running = False


class OjinAvatarService(FrameProcessor):
    """Ojin Avatar integration for Pipecat.

    This class provides integration between Ojin avatars and the Pipecat framework.
    It is abstracted from the transport layer, processing frames without caring about their source,
    but provides a room ID for use with services like Daily or LiveKit.
    """

    def __init__(
        self,
        settings: OjinAvatarSettings,
    ) -> None:
        super().__init__()

        # Use provided settings or create default settings
        self._settings = settings

        # Generate a UUID if avatar_id is not provided
        assert self._settings.ojin_avatar_id is not None

        self._audio_task: Optional[asyncio.Task] = None

        self._interaction: Optional[OjinAvatarInteraction] = None

        self._resampler = create_default_resampler()
        # For video processing
        self._auth_token: Optional[str] = None

        logger.info(
            f"OjinAvatarService initialized with avatar ID: "
            f"{self._settings.ojin_avatar_id}"
        )

    async def push_ojin_frame(self, message: OjinAvatarMessage):
        await self.push_frame(TransportMessageFrame(message=message))

    async def push_urgent_ojin_frame(self, message: OjinAvatarMessage):
        await self.push_frame(TransportMessageUrgentFrame(message=message))

    async def _start(self):
        await self.push_ojin_frame(
            AuthenticationMessage(api_key=self._settings.api_key)
        )
        # Create tasks to process audio and video
        self._audio_task = self.create_task(self._process_queued_audio())

    async def _handle_transport_message(self, frame: OjinAvatarInputFrame):
        """Process incoming messages."""
        message = frame.message
        if isinstance(message, InteractionResponseMessage):
            logger.info(f"Video frame received: {message.frame_idx}")
            # Create and push the image frame
            image_frame = OutputImageRawFrame(
                image=message.payload,
                size=self._settings.image_size,
                format="RGB",
            )
            await self.push_frame(image_frame)

            if message.is_final_response:
                logger.info("No more video frames expected")
                self._close_interaction()

        elif isinstance(message, AuthenticationResponseMessage):
            self._auth_token = message.token

        elif isinstance(message, InteractionReadyMessage):
            assert (
                self._interaction is not None
                and self._interaction.avatar_id == message.avatar_id
            )
            self._interaction.interaction_id = message.interaction_id
            self._interaction.is_running = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._start()
        elif isinstance(frame, TTSStartedFrame):
            await self._start_interaction()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSStoppedFrame):
            await self._end_interaction()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSAudioRawFrame):
            # Handle incoming audio frames
            # For example, this could be speech that needs to be processed by the avatar
            await self._handle_input_audio(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartInterruptionFrame):
            # Clear any pending audio/video data
            await self._interrupt()
            await self.push_frame(frame, direction)
        elif isinstance(frame, OjinAvatarInputFrame):
            await self._handle_transport_message(frame)
        else:
            # Pass through any other frames
            await self.push_frame(frame, direction)

    async def _interrupt(self):
        if self._interaction and self._interaction.is_running:
            logger.info("Interrupting interaction")
            await self.push_urgent_ojin_frame(
                InteractionInputMessage(interrupt=True, token=self._auth_token)
            )
            self._close_interaction()

    async def _start_interaction(self):
        self._interaction = OjinAvatarInteraction(
            avatar_id=self._settings.ojin_avatar_id,
        )
        await self.push_ojin_frame(
            StartInteractionMessage(
                avatar_id=self._settings.ojin_avatar_id, token=self._auth_token
            )
        )

    async def _end_interaction(self):
        self._interaction.is_ending = True
        # we push empty audio to signal the end of the interaction
        await self._interaction.audio_input_queue.put(
            InteractionInputMessage(is_last_input=True, token=self._auth_token)
        )

    async def _handle_input_audio(self, frame: TTSAudioRawFrame):
        """Handle incoming audio frames.

        Resamples the audio to the target sample rate and either sends it directly
        to the backend if an interaction is running, or queues it for later processing.
        """
        resampled_audio = await self._resampler.resample(
            frame.audio, frame.sample_rate, self._settings.sample_rate
        )
        assert self._interaction is not None
        # Queue the audio for later processing
        await self._interaction.audio_input_queue.put(
            InteractionInputMessage(audio=resampled_audio, token=self._auth_token)
        )
        logger.debug(
            f"Queued audio for later processing. Queue size: {self._interaction.audio_input_queue.qsize()}"
        )

    def _close_interaction(self):
        # Clear the interaction queue if it exists
        if self._interaction is not None:
            self._interaction.close()
            self._interaction = None

    async def _process_queued_audio(self):
        """Process audio that was queued before an interaction was ready."""
        while True:
            # Wait until we have a running interaction
            if not self._interaction or not self._interaction.is_running:
                # No interaction or not running yet, sleep briefly and check again
                await asyncio.sleep(0.01)
                continue

            # Get audio from the queue
            if not self._interaction.audio_input_queue.empty():
                message: InteractionInputMessage = (
                    await self._interaction.audio_input_queue.get()
                )

                if (
                    self._interaction.audio_input_queue.qsize() == 0
                    and self._interaction.is_ending
                ):
                    message.is_last_input = True

                # TODO(@JM): Batch audio messages?
                # Send to backend
                await self.push_urgent_ojin_frame(message)
                self._interaction.audio_input_queue.task_done()
            else:
                # No audio in queue, sleep briefly
                await asyncio.sleep(0.1)

    async def _stop(self):
        # Cancel queued audio processing task
        if self._audio_task:
            await self.cancel_task(self._audio_task)
            self._audio_task = None

        # Clear all buffers
        await self._interrupt()

        logger.info(f"OjinAvatarService {self._settings.ojin_avatar_id} stopped")
