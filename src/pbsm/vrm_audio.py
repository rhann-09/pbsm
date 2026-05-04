#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Based SMPL-X Modelling (PBSM)

Standalone WebSocket Audio Streaming Server for Speaches.ai for TTS and LM Studio for LLM inference.
"""

# %% Imports

# Standard Library Imports
import re
import asyncio
import threading
from typing import Any, Generator, Set

# Third-Party Imports
import websockets
import lmstudio as lms
from openai import OpenAI

# %% Classes

class VoiceManager:
    """
    Manages TTS generation by communicating with a local Speaches server.

    This class acts as a client for an OpenAI-compatible audio API (Speaches),
    chunking incoming text streams by sentence and requesting synthesized
    audio in WAV format.

    Attributes
    ----------
    model_id : str
        The identifier for the TTS model being used (e.g., Kokoro).
    voice_id : str
        The specific voice profile identifier.
    sentence_end_regex : re.Pattern
        A compiled regular expression used to chunk incoming text streams
        based on sentence-ending punctuation.
    client : openai.OpenAI
        The OpenAI client configured to point to the local Speaches server.

    Methods
    -------
    generate_audio_chunks(llm_token_generator)
        Consumes a stream of text tokens and yields synthesized audio as WAV bytes.
    _synthesize_chunk(text)
        Internal helper that requests audio from the Speaches API and returns bytes.
    """
    def __init__(
            self,
            base_url: str = "http://localhost:8000/v1",
            model_id: str = "speaches-ai/Kokoro-82M-v1.0-ONNX",
            voice_id: str = "af_heart",
            sentence_end: str = ".?!,;:",
            ) -> None:
        """
        Initializes the VoiceManager and connects to the Speaches API.

        Parameters
        ----------
        base_url : str, optional
            The base URL of the local Speaches API server
            (default is "http://localhost:8000/v1").
        model_id : str, optional
            The model identifier to use on the server
            (default is "speaches-ai/Kokoro-82M-v1.0-ONNX").
        voice_id : str, optional
            The requested voice embedding/profile (default is "af_heart").
        sentence_end : str, optional
            A string containing characters that should trigger the end of
            a spoken chunk (default is ".?!,;:").
        """
        self.model_id = model_id
        self.voice_id = voice_id
        self.sentence_end_regex = re.compile(f'([{sentence_end}])')

        # Initialize the OpenAI client to point to the local Speaches server
        self.client = OpenAI(base_url=base_url, api_key="cant-be-empty")
        print(f"Connected to Speaches TTS using model: {self.model_id}")


    def generate_audio_chunks(self, llm_token_generator: Any) -> Generator[bytes, None, None]:
        """
        Consumes a text generator, requests audio from Speaches, and yields WAV bytes.

        Iterates over a stream of text tokens, buffers them until a sentence-ending
        punctuation mark is found, requests the audio from the API, and yields
        the resulting binary data.

        Parameters
        ----------
        llm_token_generator : Any
            A generator or iterable yielding text tokens (e.g., from an LLM).

        Yields
        ------
        bytes
            The synthesized audio chunk encoded as raw WAV bytes.

        Methods Used
        ------------
        * _synthesize_chunk()
        """
        buffer = ""
        for token in llm_token_generator:
            buffer += token
            if self.sentence_end_regex.search(buffer):
                sentence = buffer.strip()
                if sentence:
                    yield self._synthesize_chunk(sentence)
                buffer = ""

        final_sentence = buffer.strip()
        if final_sentence:
            yield self._synthesize_chunk(final_sentence)


    def _synthesize_chunk(self, text: str) -> bytes:
        """
        Sends the text segment to the Speaches server and returns raw WAV bytes.

        Parameters
        ----------
        text : str
            The text segment to be synthesized into speech.

        Returns
        -------
        bytes
            The synthesized audio waveform encoded in a WAV format byte stream.
        """
        response = self.client.audio.speech.create(
            model=self.model_id,
            voice=self.voice_id,
            input=text,
            response_format="wav" # Requests raw WAV bytes to match Three.js setup
            )

        return response.content


class AudioBridge:
    """
    Manages a background WebSocket server to stream audio to a Three.js frontend.

    This class runs an asynchronous WebSocket server in a separate daemon thread,
    allowing the main thread to remain unblocked while dynamically handling
    incoming connections and broadcasting audio streams.

    Attributes
    ----------
    voice_manager : VoiceManager
        The initialized voice manager instance responsible for synthesizing audio.
    host : str
        The IP address the server binds to (default is "127.0.0.1").
    port : int
        The port the server listens on. A value of 0 allows the OS to assign an available port.
    clients : Set[Any]
        A set tracking all currently connected and active WebSocket clients.
    is_running : bool
        Flag indicating whether the background server thread is actively running.

    Methods
    -------
    start()
        Starts the WebSocket server in a background thread and waits for port binding.
    stream_to_clients(llm_token_generator)
        Generates audio chunks from the token stream and broadcasts to all clients.
    _run_server()
        Internal method that executes the core asyncio event loop.
    _handler(websocket, *args, **kwargs)
        Internal coroutine that manages individual incoming browser connections.
    _broadcast(data)
        Internal coroutine that pushes binary data to all connected sockets.
    """

    def __init__(self, voice_manager: VoiceManager, host: str = "127.0.0.1", port: int = 0) -> None:
        """
        Initializes the AudioBridge with the provided VoiceManager.

        Parameters
        ----------
        voice_manager : VoiceManager
            An initialized VoiceManager instance to handle TTS generation.
        host : str, optional
            The host IP address to bind the WebSocket server (default is "127.0.0.1").
        port : int, optional
            The port to bind to. Set to 0 to auto-assign a free port (default is 0).
        """
        self.voice_manager = voice_manager
        self.host = host
        self.port = port
        self.clients: Set[Any] = set()
        self._loop = None
        self._thread = None
        self.is_running = False

        # Use an Event to safely pass the port back to the main thread
        self._server_ready = threading.Event()

    def start(self) -> None:
        """
        Starts the WebSocket server in a background daemon thread.

        This method spawns the thread and blocks the main thread momentarily
        until the asynchronous server has successfully bound to an OS port.

        Methods Used
        ------------
        * _run_server()
        """
        self.is_running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

        # Wait for the async thread to actually bind the port before continuing
        self._server_ready.wait()
        print(f"AudioBridge WebSocket server running on ws://{self.host}:{self.port}")

    def _run_server(self) -> None:
        """
        The core event loop for the background thread.

        Initializes a new asyncio event loop, starts the WebSocket server,
        retrieves the dynamically assigned port, and signals the main thread
        to continue execution.

        Methods Used
        ------------
        * _handler()
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def start_websocket_server():
            # websockets.serve must be called inside an active async context
            async with websockets.serve(self._handler, self.host, self.port) as server:
                # Retrieve the dynamically assigned port from the OS
                self.port = server.sockets[0].getsockname()[1]

                # Signal the main thread that it can continue launching the browser
                self._server_ready.set()

                await asyncio.Future()  # Keeps the server running forever

        try:
            self._loop.run_until_complete(start_websocket_server())
        except Exception as e:
            print(f"WebSocket server stopped: {e}")
        finally:
            self._loop.close()

    async def _handler(self, websocket: Any, *args, **kwargs) -> None:
        """
        Handles incoming browser connections.

        Adds the newly connected client to the active clients set, keeps the
        connection alive, and safely removes the client upon disconnection.

        Parameters
        ----------
        websocket : Any
            The active WebSocket connection object.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        self.clients.add(websocket)
        print(f"Frontend client connected. Total clients: {len(self.clients)}")
        try:
            # Keep the handler alive until the connection closes
            await websocket.wait_closed()
        except Exception:
            pass
        finally:
            self.clients.remove(websocket)
            print("Frontend client disconnected.")

    def stream_to_clients(self, llm_token_generator: Any) -> None:
        """
        Generates audio and streams it live to all connected Three.js clients.

        Iterates through the provided LLM text generator, synthesizes the text
        into WAV byte chunks using the VoiceManager, and asynchronously dispatches
        the chunks to all connected WebSockets.

        Parameters
        ----------
        llm_token_generator : Any
            A generator or iterable yielding text tokens to be synthesized.

        Methods Used
        ------------
        * _broadcast()
        """
        if not self.clients:
            print("No clients connected to AudioBridge. Skipping audio generation.")
            return

        for wav_bytes in self.voice_manager.generate_audio_chunks(llm_token_generator):
            print(f"Synthesized sentence, sending {len(wav_bytes)} bytes to Three.js...")
            # Thread-safe dispatch to the async loop
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(self._broadcast(wav_bytes), self._loop)

    async def _broadcast(self, data: bytes) -> None:
        """
        Pushes bytes to all connected WebSockets.

        Parameters
        ----------
        data : bytes
            The raw audio byte data to be transmitted to the clients.
        """
        if self.clients:
            await asyncio.gather(*(client.send(data) for client in self.clients))


# %% Functions

def lmstudio_stream(prompt: str) -> Generator[str, None, None]:
    """
    Streams tokens from an LM Studio local server, filtering out reasoning blocks.

    Connects automatically to the running local server, submits the prompt,
    and yields text tokens while safely buffering and stripping any text
    generated inside reasoning tags.

    Note: This function is currently hardcoded to filter out the `<|channel>thought`
    tags specific to the Gemma 4 architecture. Future iterations will generalize
    this to support standard `<think>` tags and other model-specific structures.

    Parameters
    ----------
    prompt : str
        The user input prompt to send to the local LLM.

    Yields
    ------
    str
        The next generated text token from the LLM, excluding reasoning tokens.
    """
    try:
        model = lms.llm()
        prediction_stream = model.respond_stream(
            prompt,
            config={"temperature": 0.7}
        )

        in_thought = False
        buffer = ""

        for fragment in prediction_stream:
            if not fragment.content:
                continue

            buffer += fragment.content

            if not in_thought:
                # 1. Detect entering the thought process (Gemma 4 specific)
                if "<|channel>thought" in buffer:
                    pre_thought, post_thought = buffer.split("<|channel>thought", 1)
                    if pre_thought:
                        print(pre_thought, end="", flush=True)
                        yield pre_thought

                    in_thought = True
                    buffer = post_thought
                else:
                    # 2. Hold back fragments starting with '<' just in case a tag is forming
                    if "<" in buffer:
                        safe_idx = buffer.rfind("<")
                        safe_part = buffer[:safe_idx]
                        if safe_part:
                            print(safe_part, end="", flush=True)
                            yield safe_part
                        # Keep the unresolved '<...' part in the buffer
                        buffer = buffer[safe_idx:]
                    else:
                        print(buffer, end="", flush=True)
                        yield buffer
                        buffer = ""
            else:
                # 3. Detect exiting the thought process (Gemma 4 specific)
                if "channel|>" in buffer:
                    in_thought = False
                    buffer = buffer.split("channel|>", 1)[1]
                else:
                    # 4. Discard the thought text. Keep only the last 15 characters
                    # in case the 'channel|>' exit tag is split across two chunks.
                    if len(buffer) > 15:
                        buffer = buffer[-15:]

        # Flush anything remaining in the buffer when the stream ends
        if not in_thought and buffer:
            # Strip any dangling '<' characters just in case the stream ended abruptly
            buffer = buffer.replace("<", "")
            if buffer:
                print(buffer, end="", flush=True)
                yield buffer

        print("\n")

    except Exception as e:
        print(f"Error communicating with LM Studio SDK: {e}")
