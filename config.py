import os

from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://scyu-1027-resource.services.ai.azure.com/openai/v1",
)
LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-5.6-luna")
EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
)
USE_ENTRA = os.getenv("AZURE_OPENAI_USE_ENTRA", "true").lower() in (
    "1",
    "true",
    "yes",
)
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
EXA_MAX_RESULTS = int(os.getenv("EXA_MAX_RESULTS", "5"))


def build_client():
    """Build an OpenAI client for the Azure Foundry /openai/v1 endpoint.

    Entra ID (default): uses DefaultAzureCredential, so run `az login` first.
    API key fallback: set AZURE_OPENAI_USE_ENTRA=false and AZURE_OPENAI_API_KEY.
    """
    from openai import OpenAI

    if USE_ENTRA:
        from azure.identity import (
            DefaultAzureCredential,
            get_bearer_token_provider,
        )

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://ai.azure.com/.default"
        )
        return OpenAI(base_url=AZURE_OPENAI_ENDPOINT, api_key=token_provider)

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "AZURE_OPENAI_API_KEY is not set and AZURE_OPENAI_USE_ENTRA is false."
        )
    return OpenAI(base_url=AZURE_OPENAI_ENDPOINT, api_key=api_key)
