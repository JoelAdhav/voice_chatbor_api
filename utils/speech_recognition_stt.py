import speech_recognition as sr
from pydub import AudioSegment
import io
import os
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure ffmpeg/ffprobe are accessible if not in PATH
# Example: AudioSegment.converter = "/path/to/ffmpeg"
# Example: AudioSegment.ffprobe = "/path/to/ffprobe"

def convert_audio_to_text(audio_file_path: str, language: str = "en-US") -> str | None:
    """
    Converts an audio file to text using Google Web Speech API via SpeechRecognition.
    Handles various audio formats by converting them to WAV first using pydub.

    Args:
        audio_file_path: The path to the input audio file.
        language: The language code for speech recognition (e.g., "en-US", "hi-IN").

    Returns:
        The transcribed text as a string, or None if transcription fails.
    """
    recognizer = sr.Recognizer()
    text = None
    temp_wav_fd = None
    temp_wav_path = None

    try:
        # Load audio file using pydub (handles various formats)
        logger.info(f"Loading audio file: {audio_file_path}")
        audio = AudioSegment.from_file(audio_file_path)
        logger.info("Audio file loaded successfully.")

        # Create a temporary WAV file because SpeechRecognition works best with WAV
        temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_wav_fd) # Close the file descriptor, pydub will handle the file
        logger.info(f"Exporting audio to temporary WAV file: {temp_wav_path}")
        audio.export(temp_wav_path, format="wav")
        logger.info("Audio exported to WAV successfully.")

        # Use the temporary WAV file with SpeechRecognition
        with sr.AudioFile(temp_wav_path) as source:
            logger.info("Adjusting for ambient noise...")
            # recognizer.adjust_for_ambient_noise(source) # Optional: Adjust for noise
            logger.info("Listening to audio file...")
            audio_data = recognizer.record(source)
            logger.info("Audio data recorded.")

        # Recognize speech using Google Web Speech API with the specified language
        logger.info(f"Attempting speech recognition for language: {language}...")
        text = recognizer.recognize_google(audio_data, language=language)
        logger.info(f"Speech recognized successfully: {text}")

    except sr.UnknownValueError:
        logger.error(f"Google Web Speech API could not understand the audio for language {language}.")
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Web Speech API; {e}")
    except FileNotFoundError:
        logger.error(f"Audio file not found at path: {audio_file_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during audio processing: {e}", exc_info=True)
    finally:
        # Clean up the temporary WAV file
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
                logger.info(f"Temporary WAV file removed: {temp_wav_path}")
            except Exception as e:
                logger.error(f"Error removing temporary file {temp_wav_path}: {e}")

    return text

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Create a dummy audio file for testing (requires ffmpeg installed)
    # You should replace this with a real audio file path for actual testing
    test_file = "test_audio.mp3" # Or .wav, .ogg, etc.
    if not os.path.exists(test_file):
        logger.warning(f"Test audio file '{test_file}' not found. Skipping example usage.")
    else:
        # Test with English
        logger.info(f"Testing with audio file: {test_file} (Language: en-US)")
        transcribed_text_en = convert_audio_to_text(test_file, language="en-US")
        if transcribed_text_en:
            print(f"\nTranscription Result (en-US):\n{transcribed_text_en}")
        else:
            print("\nTranscription failed (en-US).")

        # Example test with Hindi (requires appropriate audio file)
        # logger.info(f"Testing with audio file: {test_file} (Language: hi-IN)")
        # transcribed_text_hi = convert_audio_to_text(test_file, language="hi-IN")
        # if transcribed_text_hi:
        #     print(f"\nTranscription Result (hi-IN):\n{transcribed_text_hi}")
        # else:
        #     print("\nTranscription failed (hi-IN).")
            print("\nTranscription failed.")
