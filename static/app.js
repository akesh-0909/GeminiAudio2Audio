const toggleButton = document.getElementById('toggle-button');
const statusIndicator = document.getElementById('status');
const visualizer = document.getElementById('visualizer');

let websocket;
let mediaStream;
let audioContext;
let scriptProcessor;
let mediaStreamSource;

const setupWebSocket = () => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/audio`;

    websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
        console.log('WebSocket connection established');
        statusIndicator.classList.add('connected');
    };

    websocket.onclose = () => {
        console.log('WebSocket connection closed');
        statusIndicator.classList.remove('connected');
        stopListening();
    };

    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        statusIndicator.classList.remove('connected');
    };

    websocket.onmessage = (event) => {
        console.log(`Received audio data from server: ${event.data.size} bytes`);
        playAudio(event.data);
    };
};

const playAudio = async (audioData) => {
    console.log("playAudio called");
    try {
        const arrayBuffer = await audioData.arrayBuffer();
        console.log("Decoding audio data...");
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.start();
        console.log("Audio playback started.");
    } catch (error) {
        console.error("Error decoding or playing audio:", error);
    }
};

const float32To16BitPCM = (input) => {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
        const s = Math.max(-1, Math.min(1, input[i]));
        output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return output;
};


const startListening = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Your browser does not support microphone access or is not running in a secure context (HTTPS).');
        return;
    }

    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
        setupWebSocket();
    }

    if (mediaStream) {
        return;
    }

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        mediaStreamSource = audioContext.createMediaStreamSource(mediaStream);

        mediaStreamSource.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        scriptProcessor.onaudioprocess = (event) => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                const inputData = event.inputBuffer.getChannelData(0);
                const pcmData = float32To16BitPCM(inputData);
                websocket.send(pcmData.buffer);
            }
        };

        toggleButton.textContent = 'Stop Listening';
        toggleButton.classList.add('active');

        setupVisualizer();

    } catch (error)
        {
        console.error('Error accessing microphone:', error);
    }
};

const stopListening = () => {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor = null;
    }
    if(mediaStreamSource){
        mediaStreamSource.disconnect();
        mediaStreamSource = null;
    }
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close();
    }
    if (websocket && (websocket.readyState === WebSocket.OPEN || websocket.readyState === WebSocket.CONNECTING)) {
        websocket.close();
    }


    toggleButton.textContent = 'Start Listening';
    toggleButton.classList.remove('active');
};

toggleButton.addEventListener('click', () => {
    if (mediaStream) {
        stopListening();
    } else {
        startListening();
    }
});

// Basic audio visualizer
const canvasCtx = visualizer.getContext('2d');
let analyser;

function visualize() {
    if (!analyser) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
        requestAnimationFrame(draw);

        analyser.getByteTimeDomainData(dataArray);

        canvasCtx.fillStyle = '#282828';
        canvasCtx.fillRect(0, 0, visualizer.width, visualizer.height);

        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = '#1DB954';

        canvasCtx.beginPath();

        const sliceWidth = visualizer.width * 1.0 / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * visualizer.height / 2;

            if (i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        canvasCtx.lineTo(visualizer.width, visualizer.height / 2);
        canvasCtx.stroke();
    };

    draw();
}

const setupVisualizer = () => {
    if (!audioContext) return;
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;

    if (mediaStreamSource) {
        mediaStreamSource.connect(analyser);
    }

    visualize();
};
