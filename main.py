from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse # Changed from StreamingResponse
from dotenv import load_dotenv
import os
import tempfile
import logging
import json
import base64 # Added for Base64 encoding
from typing import List, Dict, Optional

# Import utility functions
from utils.speech_recognition_stt import convert_audio_to_text
from utils.gemini_api import generate_response_with_gemini
from utils.elevenlabs_tts import convert_text_to_speech_stream

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Voice Chatbot API")

@app.get("/")
async def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "Welcome to the Voice Chatbot API!"}

# --- Voice Chat Endpoint ---
@app.post("/chat/voice")
async def chat_voice_endpoint(
    audio_file: UploadFile = File(...),
    history_json: Optional[str] = Form(None), # Accept history as JSON string
    language_code: str = Form("en-US") # Add language code parameter
):
    """
    Handles voice chat interaction:
    1. Receives audio file and optional conversation history.
    2. Transcribes audio to text.
    3. Generates a text response using Gemini (with history).
    4. Converts the text response to speech audio using ElevenLabs.
    5. Streams the audio response back.
    """
    temp_audio_path = None
    try:
        # 1. Save uploaded audio file temporarily
        suffix = os.path.splitext(audio_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            content = await audio_file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        logger.info(f"Temporary audio file saved at: {temp_audio_path}")
        logger.info(f"Received language code: {language_code}")

        # 2. Transcribe audio to text using the provided language code
        user_text = convert_audio_to_text(temp_audio_path, language=language_code)
        if not user_text:
            logger.error(f"Transcription failed for language: {language_code}.")
            raise HTTPException(status_code=400, detail="Failed to transcribe audio.")
        logger.info(f"Transcription successful: '{user_text}'")

        # 3. Parse conversation history if provided
        history: List[Dict[str, str]] = []
        if history_json:
            try:
                history = json.loads(history_json)
                # Basic validation (optional but recommended)
                if not isinstance(history, list) or not all(isinstance(item, dict) and 'role' in item and 'parts' in item for item in history):
                    raise ValueError("Invalid history format.")
                logger.info(f"Parsed conversation history ({len(history)} items).")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse history JSON: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid history format: {e}")

        # 4. Generate text response using Gemini
        bot_text_response = generate_response_with_gemini(user_text, history=history)
        if not bot_text_response:
            logger.error("Gemini response generation failed.")
            raise HTTPException(status_code=500, detail="Failed to generate response from language model.")
        logger.info(f"Gemini response generated: '{bot_text_response[:100]}...'")

        # 5. Convert text response to speech audio
        audio_stream_bytes = convert_text_to_speech_stream(bot_text_response)
        if not audio_stream_bytes:
            logger.error("Text-to-speech conversion failed.")
            raise HTTPException(status_code=500, detail="Failed to convert response to speech.")
        logger.info("Text-to-speech conversion successful.")

        # 6. Encode audio bytes to Base64
        audio_base64 = base64.b64encode(audio_stream_bytes).decode('utf-8')
        logger.info("Audio encoded to Base64.")

        # 7. Return JSON response with transcription, bot text, and Base64 audio
        return JSONResponse(content={
            "user_transcription": user_text,
            "bot_response_text": bot_text_response,
            "bot_response_audio": audio_base64 # Send Base64 encoded audio
        })

    except HTTPException as http_exc:
        # Re-raise HTTPException to let FastAPI handle it
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred in /chat/voice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    finally:
        # Clean up the temporary audio file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.info(f"Temporary audio file removed: {temp_audio_path}")
            except Exception as e:
                 logger.error(f"Error removing temporary file {temp_audio_path}: {e}")


if __name__ == "__main__":
    import uvicorn
    # Render provides the PORT environment variable
    port = int(os.getenv("PORT", 8000)) # Keep default for local testing if needed
    # Use host="0.0.0.0" to bind to all interfaces, disable reload for production
    # Render will use the start command defined in its dashboard, but this is good practice
    uvicorn.run("main:app", host="0.0.0.0", port=port) # Removed reload=True
