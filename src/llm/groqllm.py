import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq


class GroqLLM:

    def __init__(self, user_input_control):
        self.user_input_control = user_input_control

    def get_llm_model(self):
        try:
            # Load environment variables
            load_dotenv()

            # Get API key from .env
            groq_api_key = os.getenv("GROQ_API_KEY")

            # Get selected model from user input
            selected_model = self.user_input_control.get(
                "selected_model",
                "openai/gpt-oss-20b"
            )

            # Validate inputs
            if not groq_api_key:
                raise ValueError("Groq API key is missing.")

            if not selected_model:
                raise ValueError("Please select an LLM model.")

            # Initialize Groq LLM
            llm = ChatGroq(
                api_key=groq_api_key,
                model=selected_model
            )

            return llm

        except Exception as e:
            raise ValueError(f"Error initializing Groq LLM: {e}") from e