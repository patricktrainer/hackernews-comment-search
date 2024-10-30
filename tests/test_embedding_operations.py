import pytest
from src.embedding_operations import EmbeddingOperations
from src.connection import open_connection
from src.operations import EmbeddingKey

@pytest.fixture
def setup_db():
    con = open_connection(":memory:")
    con.execute("CREATE TABLE embeddings (text VARCHAR, model VARCHAR, embedding DOUBLE[])")
    yield con
    con.close()

def test_pickle_embeddings(setup_db):
    con = setup_db
    embedding_ops = EmbeddingOperations(con, "test_cache.pkl")
    texts = ["test text 1", "test text 2"]
    model = "test-model"
    embeddings = embedding_ops.pickle_embeddings(texts, model)
    assert len(embeddings) == 2

def test_duckdb_embeddings(setup_db):
    con = setup_db
    embedding_ops = EmbeddingOperations(con, "test_cache.pkl")
    texts = ["test text 1", "test text 2"]
    model = "test-model"
    embeddings = embedding_ops.duckdb_embeddings(texts, model)
    assert len(embeddings) == 2

def test_cosine_similarity(setup_db):
    con = setup_db
    embedding_ops = EmbeddingOperations(con, "test_cache.pkl")
    l1 = [1.0, 2.0, 3.0]
    l2 = [1.0, 2.0, 3.0]
    similarity = embedding_ops.cosine_similarity(l1, l2)
    assert similarity == 1.0

def test_get_similarity(setup_db):
    con = setup_db
    embedding_ops = EmbeddingOperations(con, "test_cache.pkl")
    text = "test text"
    model = "test-model"
    con.execute("INSERT INTO embeddings VALUES (?, ?, ?)", [text, model, [1.0, 2.0, 3.0]])
    result = embedding_ops.get_similarity(text, model)
    assert len(result) == 0
