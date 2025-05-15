"""Test the Ojin Avatar service integration with Pipecat."""

import asyncio
import os
import sys
import unittest
from unittest import TestCase

from loguru import logger

# Get the absolute path to the project root and add it to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

# Import pipecat modules
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.ojin.ojin_avatar import OjinAvatarService, OjinAvatarSettings
from pipecat.services.ojin.ojin_avatar_messages import (
    AuthenticationResponseMessage,
    InteractionReadyMessage,
    InteractionResponseMessage,
)

# Import ojin modules
from tests.mock.mock_ojin_avatar_transport import (
    MockOjinAvatarClient,
    MockOjinAvatarTransport,
)
from tests.mock.mock_tts import MockTTSProcessor


class TestOjinAvatarService(TestCase):
    """Test the OjinAvatarService integration with Pipecat."""

    async def asyncSetUp(self):
        """Set up the test environment."""
        # Create a mock client with message counters
        self.client = MockOjinAvatarClient()

        # Create the transport with our client
        self.transport = MockOjinAvatarTransport(self.client)

        # Create the TTS processor with test audio
        self.tts = MockTTSProcessor(
            {
                "audio_sequence": [
                    ("./tests/assets/hello.wav", 3),
                    ("./tests/assets/hello.wav", 6),
                ],
                "event_sequence": [
                    ("user_started_speaking", 1),
                    ("user_stopped_speaking", 2),
                    ("user_started_speaking", 4),
                    ("user_stopped_speaking", 5),
                ],
                "chunk_size": 4096,
                "chunk_delay": 0.2,
            },
        )

        # Create the avatar service
        self.avatar = OjinAvatarService(
            OjinAvatarSettings(
                api_key="test",
                ojin_avatar_id="test",
                sample_rate=16000,
                image_size=(1920, 1080),
            ),
        )

        # Create the pipeline
        self.pipeline = Pipeline(
            [self.transport.input(), self.tts, self.avatar, self.transport.output()]
        )

        # Create the pipeline task
        self.task = PipelineTask(
            pipeline=self.pipeline,
            params=PipelineParams(
                allow_interruptions=True, enable_metrics=True, enable_usage_metrics=True
            ),
        )

        # Create the runner
        self.runner = PipelineRunner()

    def setUp(self):
        """Set up the test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.asyncSetUp())

    def tearDown(self):
        """Clean up after the test."""
        self.loop.close()

    async def run_pipeline(self, duration=5.0):
        """Run the pipeline for a specified duration."""
        # Start the pipeline
        run_task = asyncio.create_task(self.runner.run(self.task))

        # Wait for the specified duration
        await asyncio.sleep(duration)

        # Cancel the pipeline task
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

    def test_message_responses(self):
        """Test that the expected messages are received."""
        # Run the pipeline for 5 seconds
        self.loop.run_until_complete(self.run_pipeline(10.0))

        # Print summary of collected messages
        logger.info("Message counts:")
        logger.info(
            f"- Authentication responses: {self.client.messages_received[AuthenticationResponseMessage]}"
        )
        logger.info(
            f"- Interaction ready messages: {self.client.messages_received[InteractionReadyMessage]}"
        )
        logger.info(
            f"- Interaction response messages: {self.client.messages_received[InteractionResponseMessage]}"
        )

        # Verify that exactly 1 authentication response was received
        self.assertEqual(
            self.client.messages_received[AuthenticationResponseMessage],
            1,
            f"Expected 1 authentication response, got {self.client.messages_received[AuthenticationResponseMessage]}",
        )

        # Verify that exactly 2 interaction ready messages were received
        self.assertEqual(
            self.client.messages_received[InteractionReadyMessage],
            2,
            f"Expected 2 interaction ready messages, got {self.client.messages_received[InteractionReadyMessage]}",
        )

        # Verify that at least 10 interaction response messages were received
        min_expected_video_frames = 100
        self.assertGreaterEqual(
            self.client.messages_received[InteractionResponseMessage],
            min_expected_video_frames,
            f"Expected at least {min_expected_video_frames} interaction response messages, got {self.client.messages_received[InteractionResponseMessage]}",
        )


if __name__ == "__main__":
    unittest.main()
