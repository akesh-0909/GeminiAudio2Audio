# Gemini Audio to Audio

This project is an audio-to-audio pipeline that uses the Gemini API to generate audio in real-time. It provides a web interface to stream audio from your microphone to the Gemini API and play back the generated audio.

## Setup

1.  **Create a virtual environment:**

    It is recommended to use a virtual environment to install the dependencies. If you are using `uv`, you can create a virtual environment with the following command:

    ```bash
    uv venv
    ```

2.  **Install the dependencies:**

    Install the dependencies using `uv`:

    ```bash
    uv pip install -e .
    ```

3.  **Set up your Gemini API key:**

    Create a `.env` file in the root of the project and add your Gemini API key to it:

    ```
    GEMINI_API_KEY=your_api_key
    ```

## Running the Application

To run the application, activate your virtual environment (if you created one) and then execute the following command:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

This will start the FastAPI server. You can then open your browser and navigate to `http://localhost:8000/static/index.html` to use the application.
# GeminiAudio2Audio
