import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("Gemini API key not found in environment variables.")
    # Depending on the application structure, you might want to raise an error
    # or handle this in a way that prevents the app from starting without the key.
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}", exc_info=True)
        # Handle configuration error appropriately

# --- Configuration for the Generative Model ---
# See https://ai.google.dev/docs/concepts#model_parameters for options
GENERATION_CONFIG = {
    "temperature": 0.7, # Controls randomness. Lower for more predictable, higher for more creative.
    "top_p": 1.0,       # Nucleus sampling parameter
    "top_k": 1,         # Top-k sampling parameter
    "max_output_tokens": 2048, # Maximum length of the response
}

# --- Safety Settings ---
# See https://ai.google.dev/docs/concepts#safety_settings
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
# --- Model Selection ---
# Use a model suitable for chat, like 'gemini-pro' or 'gemini-1.5-flash' etc.
MODEL_NAME = "gemini-1.5-flash" # Or "gemini-pro" or other available models

def generate_response_with_gemini(user_text: str, history: List[Dict[str, str]] = None) -> str | None:
    """
    Generates a response using the Google Gemini API based on user input and conversation history.

    Args:
        user_text: The latest text input from the user.
        history: A list of dictionaries representing the conversation history,
                 where each dictionary has 'role' ('user' or 'model') and 'parts' (list of strings).
                 Example: [{'role': 'user', 'parts': ['Hello']}, {'role': 'model', 'parts': ['Hi there!']}]

    Returns:
        The generated text response from Gemini, or None if generation fails.
    """
    if not GEMINI_API_KEY:
        logger.error("Cannot generate response: Gemini API key is not configured.")
        return None
    if not user_text:
        logger.warning("No user text provided for Gemini.")
        return None

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS
        )

        # Start a chat session if history is provided
        if history:
            chat = model.start_chat(history=history)
            logger.info(f"Starting chat with history ({len(history)} items). Sending message: '{user_text[:50]}...'")
            response = chat.send_message(user_text, stream=False) # Use stream=True for streaming responses
        else:
            # If no history, send a single message
            logger.info(f"Sending single message (no history): '{user_text[:50]}...'")
            response = model.generate_content(user_text, stream=False)

        # Check for safety blocks or empty response
        if not response.parts:
             if response.prompt_feedback.block_reason:
                 logger.warning(f"Response blocked due to safety settings: {response.prompt_feedback.block_reason}")
                 return f"I cannot respond to that due to safety guidelines ({response.prompt_feedback.block_reason})."
             else:
                 logger.warning("Gemini response was empty or blocked for unknown reasons.")
                 return "I'm sorry, I couldn't generate a response for that."

        generated_text = response.text
        logger.info(f"Gemini response received: '{generated_text[:100]}...'")
        return generated_text

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        # Attempt to get more details if available (might vary based on SDK version/error type)
        # if hasattr(e, 'response') and e.response:
        #     logger.error(f"Gemini API error details: {e.response.text}")
        return None

# Example usage (for testing purposes)
if __name__ == "__main__":
    test_query = "What is the weather like in London today?"
    logger.info(f"Testing Gemini API with query: '{test_query}'")

    # Example with history
    test_history = [
        {'role': 'user', 'parts': ['What is the capital of France?']},
        {'role': 'model', 'parts': ['The capital of France is Paris.']}
    ]
    response_with_history = generate_response_with_gemini(test_query, history=test_history)
    if response_with_history:
        print(f"\nResponse (with history):\n{response_with_history}")
    else:
        print("\nGemini API call failed (with history).")

    # Example without history
    response_without_history = generate_response_with_gemini(test_query)
    if response_without_history:
        print(f"\nResponse (without history):\n{response_without_history}")
    else:
        print("\nGemini API call failed (without history).")
