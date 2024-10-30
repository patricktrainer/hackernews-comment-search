import os
import pytest
import yaml
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def config_data():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

def test_database_name(config_data):
    assert config_data["database"]["name"] == "hn_embeddings"

def test_model_name(config_data):
    assert config_data["model"]["name"] == "text-embedding-ada-002"

def test_pickle_cache_path(config_data):
    assert config_data["paths"]["pickle_cache"] == "data/embeddings_cache.pkl"

def test_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key == "YOUR_OPENAI_API_KEY"

def test_rag_pipeline_top_k(config_data):
    assert config_data["rag_pipeline"]["top_k"] == 10
