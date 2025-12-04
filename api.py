import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from web_logging import setup_logging
from geminiAI import WebSocketAudioLoop

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount the static directory to serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Gemini Audio to Audio API"}

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket, mode: str = Query("none", enum=["camera", "screen", "none"])):
    await websocket.accept()
    logger.info(f"WebSocket connection established with mode: {mode}")
    audio_loop = WebSocketAudioLoop(websocket, video_mode=mode)
    try:
        await audio_loop.run()
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"An error occurred in the WebSocket: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
