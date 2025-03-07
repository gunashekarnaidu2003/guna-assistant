from flask import Flask, request, jsonify
from flask_cors import CORS
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import pyttsx3
import cohere
import base64
import io

app = Flask(__name__)
CORS(app)

# Initialize Cohere client
cohere_client = cohere.Client('710g5c3QeqSwVH180Xt0HMSLOLzXobcQeSQQDxfd')

# Initialize the TTS engine
engine = pyttsx3.init()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    
    try:
        # Generate response using Cohere
        response = cohere_client.chat(
            model='command-r',
            message=user_input
        )
        
        return jsonify({
            'success': True,
            'message': response.text.strip()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    try:
        # Get audio data from request
        audio_data = request.json.get('audio')
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        
        # Convert to numpy array
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Use speech recognition
        recognizer = sr.Recognizer()
        audio = sr.AudioData(audio_np.tobytes(), 16000, 2)
        text = recognizer.recognize_google(audio)
        
        return jsonify({
            'success': True,
            'text': text
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(port=5000)