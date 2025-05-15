from abc import ABC, abstractmethod
from dataclasses import dataclass

from pipecat.frames.frames import Frame


@dataclass
class OjinAvatarMessage(ABC):
    """Base class for all Ojin Avatar messages.

    All messages exchanged with the Ojin Avatar backend inherit from this class.
    """

    pass


@dataclass
class OjinAvatarInputFrame(Frame):
    message: OjinAvatarMessage


@dataclass
class OjinAvatarMessageWithToken(OjinAvatarMessage):
    """Base class for all Ojin Avatar messages.

    All messages exchanged with the Ojin Avatar backend inherit from this class.
    """

    token: str
    pass


@dataclass
class AuthenticationMessage(OjinAvatarMessage):
    """Authentication request message sent to the Ojin Avatar backend.

    Contains the API key required for authentication with the service.
    """

    api_key: str


@dataclass
class AuthenticationResponseMessage(OjinAvatarMessageWithToken):
    """Authentication response message received from the Ojin Avatar backend.

    Contains the authentication token to be used for subsequent requests.
    """


@dataclass
class StartInteractionMessage(OjinAvatarMessageWithToken):
    """Message to start a new interaction with an Ojin Avatar.

    Contains the ID of the avatar to interact with.
    """

    avatar_id: str


@dataclass
class StartInteractionResponseMessage(OjinAvatarMessage):
    """Response message for a successful interaction start.

    Contains the unique ID assigned to the new interaction.
    """

    interaction_id: str


@dataclass
class InteractionReadyMessage(OjinAvatarMessage):
    """Message indicating that an interaction is ready to begin.

    Contains the interaction ID, room ID for communication, and avatar ID.
    """

    interaction_id: str
    avatar_id: str


@dataclass
class InteractionResponseMessage(OjinAvatarMessage):
    """Response message containing video data from the avatar.

    Contains the interaction ID, avatar ID, and the video data as bytes.
    """

    interaction_id: str
    avatar_id: str
    payload: bytes
    is_final_response: bool = False
    frame_idx: int = 0


@dataclass
class InteractionInputMessage(OjinAvatarMessageWithToken):
    """Message containing audio input for the avatar.

    Contains the audio data as bytes and a flag indicating if this is the last input.
    """

    audio: bytes = None
    is_last_input: bool = False
    interrupt: bool = False


class IOjinAvatarClient(ABC):
    """Interface for Ojin Avatar client communication.

    Defines the contract for sending and receiving messages to/from the Ojin Avatar
    client.
    """

    @abstractmethod
    def send_message(self, message: OjinAvatarMessage) -> None:
        """Send a message to Ojin backend

        Args:
           message: The message to send.

        """
    
    @abstractmethod
    async def receive_message(self) -> OjinAvatarMessage:
        """Receive a message from Ojin backend.

        Returns:
            The received message.

        """

    @abstractmethod
    def close(self):
        """Close the client."""
