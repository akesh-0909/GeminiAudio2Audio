"""
## Documentation
Quickstart: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup

To install the dependencies for this script, run:

```
pip install google-genai opencv-python pyaudio pillow mss
```
"""

import os
import asyncio
import base64
import io
import traceback
import logging
import time
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("cli_audio_debug.log", mode='w'),
        logging.StreamHandler()
    ]
)
# ---

import cv2
import pyaudio
import PIL.Image
import mss

import argparse

from google import genai
from google.genai import types

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

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

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        logging.info(f"AudioLoop initialized with video_mode: {self.video_mode}")

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            logging.info(f"Sending text to Gemini: '{text}'")
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)
            logging.info("Putting video frame into out_queue.")
            await self.out_queue.put(frame)

        # Release the VideoCapture object
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
            logging.info("Putting screen capture into out_queue.")
            await self.out_queue.put(frame)

    async def send_realtime(self):
        logging.info("send_realtime task started.")
        while True:
            try:
                msg = await self.out_queue.get()
                logging.info(f"Got message from out_queue, sending to Gemini. Type: {msg.get('mime_type')}")
                await self.session.send(input=msg)
                logging.debug("Message successfully sent to Gemini.")
                self.out_queue.task_done()
            except asyncio.CancelledError:
                logging.info("send_realtime task cancelled.")
                break
            except Exception as e:
                logging.error(f"Error in send_realtime: {e}", exc_info=True)
                break

    async def listen_audio(self):
        logging.info("listen_audio task started.")
        mic_info = pya.get_default_input_device_info()
        logging.info(f"Using default input device: {mic_info['name']}")
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        logging.info("Microphone audio stream opened successfully.")

        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        
        while True:
            try:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                
                # Calculate RMS of the audio chunk
                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2)) if audio_data.size > 0 else 0
                
                logging.info(f"Read {len(data)} bytes from mic. RMS: {rms:.4f}")
                
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
                logging.debug("Put audio chunk into out_queue.")
            except asyncio.CancelledError:
                logging.info("listen_audio task cancelled.")
                break
            except Exception as e:
                logging.error(f"Error in listen_audio: {e}", exc_info=True)
                break
        
        logging.info("Closing microphone audio stream.")
        self.audio_stream.close()

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        logging.info("receive_audio task started.")
        while True:
            try:
                logging.info("Waiting for a turn from session.receive()...")
                turn = self.session.receive()
                logging.info("Received a turn generator from session.")
                async for response in turn:
                    if data := response.data:
                        logging.info(f"Received audio data from Gemini: {len(data)} bytes")
                        self.audio_in_queue.put_nowait(data)
                    if text := response.text:
                        logging.info(f"Received text from Gemini: {text}")
                        print(text, end="")

                logging.info("Finished processing turn from Gemini.")
                # If you interrupt the model, it sends a turn_complete.
                # For interruptions to work, we need to stop playback.
                # So empty out the audio queue because it may have loaded
                # much more audio than has played yet.
                if not self.audio_in_queue.empty():
                    logging.warning("Interrupt detected or turn complete. Emptying audio_in_queue.")
                    while not self.audio_in_queue.empty():
                        self.audio_in_queue.get_nowait()
                        self.audio_in_queue.task_done()
            except asyncio.CancelledError:
                logging.info("receive_audio task cancelled.")
                break
            except Exception as e:
                logging.error(f"Error in receive_audio: {e}", exc_info=True)
                break

    async def play_audio(self):
        logging.info("play_audio task started.")
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        logging.info("PyAudio output stream opened.")
        while True:
            try:
                logging.debug("Waiting for bytestream from audio_in_queue...")
                bytestream = await self.audio_in_queue.get()
                logging.info(f"Got {len(bytestream)} bytes from audio_in_queue to play.")
                await asyncio.to_thread(stream.write, bytestream)
                logging.debug("Finished writing bytestream to output.")
                self.audio_in_queue.task_done()
            except asyncio.CancelledError:
                logging.info("play_audio task cancelled.")
                break
            except Exception as e:
                logging.error(f"Error in play_audio: {e}", exc_info=True)
                break
        
        logging.info("Closing PyAudio output stream.")
        stream.close()


    async def run(self):
        logging.info("Starting AudioLoop run.")
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                logging.info("Gemini Live session established.")

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                logging.info("Initialized audio_in_queue and out_queue.")

                logging.info("Creating tasks...")
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    logging.info("Creating get_frames task for camera.")
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    logging.info("Creating get_screen task for screen sharing.")
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                logging.info("All tasks created.")

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            logging.info("Run loop cancelled, user requested exit.")
            pass
        except ExceptionGroup as EG:
            logging.error("ExceptionGroup caught in run loop.", exc_info=True)
            if getattr(self, 'audio_stream', None) and not self.audio_stream.is_stopped():
                self.audio_stream.close()
            traceback.print_exception(EG)
        finally:
            logging.info("AudioLoop run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())
