"""
## Documentation
Quickstart: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup

To install the dependencies for this script, run:

```
pip install google-genai opencv-python pillow mss
```
"""

import os
import asyncio
import base64
import io
import traceback
import wave
import logging
import json

import cv2
import PIL.Image
import mss

from google import genai
from google.genai import types
from dotenv import load_dotenv;load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHANNELS = 1

MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"

DEFAULT_MODE = "camera"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)


CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
)

def create_wav_header(sample_rate, channels, sample_width, num_frames):
    """Creates a WAV header for raw PCM data."""
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    subchunk2_size = num_frames * block_align

    header = io.BytesIO()
    with wave.open(header, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.setnframes(num_frames)

    header.seek(0)
    return header.read()

class WebSocketAudioLoop:
    def __init__(self, websocket, video_mode=DEFAULT_MODE):
        self.websocket = websocket
        self.video_mode = video_mode
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)
        self.session = None

    async def _get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]
        i = sct.grab(monitor)
        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            if self.session:
                await self.session.send(input=msg)

    async def receive_from_client(self):
        while True:
            try:
                message = await self.websocket.receive()
                if 'bytes' in message:
                    data = message['bytes']
                    logger.info(f"Received audio chunk from client: {len(data)} bytes")
                    await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
                elif 'text' in message:
                    text_data = message['text']
                    logger.info(f"Received text message from client: {text_data}")
                    try:
                        json_data = json.loads(text_data)
                        if json_data.get("type") == "end_of_turn":
                            logger.info("Client signaled end of turn.")
                            await self.session.send(end_of_turn=True)
                    except json.JSONDecodeError:
                        logger.error(f"Could not decode JSON from text message: {text_data}")

            except Exception as e:
                logger.error(f"Error receiving from client: {e}")
                break

    async def gemini_receiver(self):
        while True:
            logger.info("Gemini receiver: waiting for turn...")
            turn = self.session.receive()
            audio_buffer = bytearray()
            async for response in turn:
                if data := response.data:
                    audio_buffer.extend(data)
                if text := response.text:
                    logger.info(f"Gemini response text: {text}")
            
            if audio_buffer:
                logger.info(f"Gemini turn ended. Audio buffer size: {len(audio_buffer)} bytes")
                num_frames = len(audio_buffer) // (CHANNELS * 2) # 2 bytes for 16-bit PCM
                wav_header = create_wav_header(RECEIVE_SAMPLE_RATE, CHANNELS, 2, num_frames)
                wav_data = wav_header + audio_buffer
                logger.info(f"Sending WAV data to client: {len(wav_data)} bytes")
                await self.websocket.send_bytes(wav_data)
            else:
                logger.info("Gemini turn ended with no audio data.")

    async def run(self):
        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self.send_realtime())
                    tg.create_task(self.receive_from_client())
                    tg.create_task(self.gemini_receiver())

                    if self.video_mode == "camera":
                        tg.create_task(self.get_frames())
                    elif self.video_mode == "screen":
                        tg.create_task(self.get_screen())

        except asyncio.CancelledError:
            pass
        except Exception as e:
            traceback.print_exception(e)
