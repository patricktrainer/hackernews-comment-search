import os
import pickle
from typing import List, Tuple, Dict
from .connection import DuckDBPyConnection
from duckdb.typing import DuckDBPyType


ARRAY_TYPE = DuckDBPyType(list[float])  # type: ignore
PickleCache = Dict[Tuple[str, str], List[float]]


class EmbeddingKey:
    def __init__(self, text: str, model: str):
        self._text = text
        self._model = model

    @property
    def text(self):
        return self._text

    @property
    def model(self):
        return self._model

    def __eq__(self, other):
        if isinstance(other, EmbeddingKey):
            return self.text == other.text and self.model == other.model
        return False

    def __hash__(self):
        return hash((self.text, self.model))


def write_embedding_to_table(
    con: DuckDBPyConnection, key: EmbeddingKey, embedding: List[float]
) -> DuckDBPyConnection:
    """
    Writes the given embedding to the `embeddings` table in the database.

    Args:
        con (DuckDBPyConnection): The connection to the DuckDB database.
        key (EmbeddingKey): The key associated with the embedding.
        embedding (List[float]): The embedding vector.

    Returns:
        DuckDBPyConnection: The connection to the DuckDB database after the insertion.
    """
    create_table_if_not_exists(con)
    con.execute("INSERT INTO embeddings VALUES (?, ?, ?)", [key.text, key.model, embedding])
    return con


def create_table_if_not_exists(con) -> None:
    """
    Creates a table named `embeddings` if it doesn't already exist in the database.

    Args:
        con: The database connection object.

    Returns:
        None
    """
    con.from_query(
        f"CREATE TABLE IF NOT EXISTS embeddings (text VARCHAR, model VARCHAR, embedding {ARRAY_TYPE})"
    )


def is_key_in_table(con: DuckDBPyConnection, key: EmbeddingKey) -> bool:
    """
    Check if a key exists in the embeddings table.

    Args:
        con (DuckDBPyConnection): The connection to the DuckDB database.
        key (EmbeddingKey): The key to check.

    Returns:
        bool: True if the key exists in the table, False otherwise.
    """
    create_table_if_not_exists(con)
    result = con.execute(
        "SELECT EXISTS(SELECT * FROM embeddings WHERE text=? AND model=?)",
        [key.text, key.model],
    ).fetchone()
    if result:
        return result[0]
    return False


def list_keys_in_table(
    con: DuckDBPyConnection, keys: List[EmbeddingKey]
) -> list[EmbeddingKey]:
    """
    Returns a list of keys that exist in the specified table.

    Args:
        con (DuckDBPyConnection): The connection to the DuckDB database.
        keys (List[EmbeddingKey]): The keys to check in the table.

    Returns:
        List[EmbeddingKey]: A list of keys that exist in the table.
    """
    keys_in_table = []

    for key in keys:
        if is_key_in_table(con, key):
            keys_in_table.append(key)
    return keys_in_table


def load_pickle_cache(pickle_path: str) -> PickleCache:
    """
    Load a pickle cache from the given file path.

    Args:
        pickle_path (str): The path to the pickle file.

    Returns:
        PickleCache: The loaded pickle cache.

    """
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            return pickle.load(file)
    return {}


def write_pickle_cache_to_duckdb(con: DuckDBPyConnection, pickle_path: str) -> None:
    """
    Writes the contents of a pickle cache to a DuckDB database.

    Args:
        con (DuckDBPyConnection): The connection to the DuckDB database.
        pickle_path (str): The path to the pickle cache file.

    Returns:
        None
    """
    cache = load_pickle_cache(pickle_path)
    create_table_if_not_exists(con)
    for key, value in cache.items():
        embedding_key = EmbeddingKey(key[0], key[1])
        write_embedding_to_table(con, embedding_key, value)


# Function to save the cache to a file
def save_pickle_cache(cache: PickleCache, cache_path: str) -> None:
    """
    Save the given cache object as a pickle file.

    Args:
        cache (PickleCache): The cache object to be saved.
        cache_path (str): The path to save the pickle file.

    Returns:
        None
    """
    with open(cache_path, "wb") as file:
        pickle.dump(cache, file)


def get_embedding_from_table(con: DuckDBPyConnection, key: EmbeddingKey) -> List[float]:
    """
    Retrieves the embedding from the 'embeddings' table based on the given key.

    Args:
        con (DuckDBPyConnection): The connection to the DuckDB database.
        key (EmbeddingKey): The key to search for in the table.

    Returns:
        List[float]: The embedding associated with the given key.

    Raises:
        ValueError: If the embedding for the given key is not found in the table.
    """
    result = con.execute(
        "SELECT embedding FROM embeddings WHERE text=? AND model=?", [key.text, key.model]
    ).fetchone()
    if result:
        return result[0]
    raise ValueError(f"Embedding for {key.text} with model {key.model} not found in table")
