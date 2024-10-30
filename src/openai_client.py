import os
from dotenv import load_dotenv
from typing import List
from openai import OpenAI

load_dotenv()


def get_openai_client() -> OpenAI:
    """
    Returns an instance of the OpenAI client.

    :return: An instance of the OpenAI client.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key not found in environment variables.")
    client = OpenAI(api_key=key)
    return client


def create_embedding(
    text: str, model: str = "text-embedding-ada-002", **kwargs
) -> List[float]:
    """
    Creates an embedding for the given text using the specified model.

    Args:
        text (str): The input text to be embedded.
        model (str, optional): The name of the model to use for embedding. Defaults to "text-embedding-ada-002".
        **kwargs: Additional keyword arguments to be passed to the OpenAI client.

    Returns:
        List[float]: The embedding vector for the input text.
    """
    try:
        client = get_openai_client()
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return []

    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=model, **kwargs)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return []
