import requests
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# ElevenLabs API Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# You can change this to a preferred voice ID from your ElevenLabs account
DEFAULT_VOICE_ID = "eVItLK1UvXctxuaRV2Oq" # Example: Default voice "Rachel"
TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{DEFAULT_VOICE_ID}/stream"

def convert_text_to_speech_stream(text: str) -> bytes | None:
    """
    Converts text to speech using the ElevenLabs API and returns the audio data as bytes.

    Args:
        text: The text content to convert to speech.

    Returns:
        The audio data as bytes (e.g., MP3), or None if conversion fails.
    """
    if not ELEVENLABS_API_KEY:
        logger.error("ElevenLabs API key not found in environment variables.")
        return None
    if not text:
        logger.warning("No text provided for TTS conversion.")
        return None

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }

    # Documentation: https://elevenlabs.io/docs/api-reference/text-to-speech-stream
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2", # Or another suitable model
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0, # Adjust style exaggeration (0.0 to 1.0)
            "use_speaker_boost": True
        }
    }

    try:
        logger.info(f"Requesting TTS from ElevenLabs for text: '{text[:50]}...'")
        response = requests.post(TTS_URL, json=data, headers=headers, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Read the audio content from the streaming response
        audio_content = b""
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                audio_content += chunk

        logger.info("TTS audio stream received successfully.")
        return audio_content

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling ElevenLabs API: {e}", exc_info=True)
        # Log detailed error if available in response
        try:
            error_detail = response.json()
            logger.error(f"ElevenLabs API error detail: {error_detail}")
        except Exception:
            logger.error(f"ElevenLabs API response content: {response.content}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during TTS conversion: {e}", exc_info=True)
        return None

# Example usage (for testing purposes)
if __name__ == "__main__":
    test_text = "Hello! This is a test of the ElevenLabs text-to-speech conversion."
    logger.info(f"Testing TTS with text: '{test_text}'")
    audio_bytes = convert_text_to_speech_stream(test_text)

    if audio_bytes:
        output_filename = "test_output.mp3"
        try:
            with open(output_filename, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"TTS audio saved successfully to {output_filename}")
            print(f"\nTTS audio saved to {output_filename}. You can play this file.")
        except Exception as e:
            logger.error(f"Error saving TTS audio to file: {e}")
            print("\nFailed to save TTS audio.")
    else:
        print("\nTTS conversion failed.")
