from typing import List
import src.operations as operations
import src.openai_client as openai_client
from .connection import DuckDBPyConnection
from .operations import EmbeddingKey


# Function to get embeddings, using the cache
def pickle_embeddings(
    texts: List[str], model: str, pickle_path: str
) -> List[List[float]]:
    """
    Retrieves or creates embeddings for a list of texts using python pickle as the storage engine.

    Args:
        texts (List[str]): The list of texts for which embeddings need to be retrieved or created.
        model (str): The name of the model to be used for creating embeddings.
        pickle_path (str): The path to save the pickle cache.

    Returns:
        List[List[float]]: A list of embeddings, where each embedding is represented as a list of floats.
    """
    embeddings = []
    pickle_cache = operations.load_pickle_cache(pickle_path)

    for text in texts:
        key = EmbeddingKey(text, model)
        if key not in pickle_cache:
            pickle_cache[key] = openai_client.create_embedding(text, model=model)
        embeddings.append(pickle_cache[key])
    operations.save_pickle_cache(pickle_cache, pickle_path)
    return embeddings


# Function to get embeddings, using the cache
def duckdb_embeddings(
    texts: List[str], model: str, con: DuckDBPyConnection
) -> List[List[float]]:
    """
    Retrieves or creates embeddings for a list of texts using DuckDB as the storage engine.

    Args:
        texts (List[str]): The list of texts for which embeddings need to be retrieved or created.
        model (str): The name of the model to be used for creating embeddings.
        con (DuckDBPyConnection): The connection object for DuckDB.

    Returns:
        List[List[float]]: A list of embeddings, where each embedding is represented as a list of floats.
    """
    embeddings = []
    for text in texts:
        key = EmbeddingKey(text, model)
        # check to see if embedding is in duckdb table
        result = operations.is_key_in_table(con, key)
        if result:
            print("Embedding found in table")
            # if so, get it
            embedding = operations.get_embedding_from_table(con, key)
            embeddings.append(embedding)
        else:
            print("Embedding not found in table")
            print("Creating new embedding")
            # if not, create it
            embedding = openai_client.create_embedding(text, model)
            # and write it to the table
            operations.write_embedding_to_table(con, key, embedding)
            embeddings.append(embedding)        
    return embeddings


def cosine_similarity(con: DuckDBPyConnection, l1, l2) -> float:
    """
    Calculates the cosine similarity between two lists.

    Parameters:
    con (DuckDBPyConnection): The connection to the DuckDB database.
    l1: The first list.
    l2: The second list.

    Returns:
    float: The cosine similarity between the two lists.
    """
    return con.execute(f"SELECT list_cosine_similarity({l1}, {l2})").fetchall()[0][0]


def get_similarity(
    con: DuckDBPyConnection, text: str, model: str
) -> list[tuple[str, float]]:
    """
    Calculates the cosine similarity between the input text and all other texts in the table.

    Args:
        con (DuckDBPyConnection): The connection to the DuckDB database.
        text (str): The input text for which to calculate the similarity.
        model (str): The name of the embedding model to use.

    Returns:
        list[tuple[str, float]]: A list of tuples containing the text and its similarity score, sorted in descending order of similarity.
    """
    sql = """
        WITH q1 AS (
            SELECT 
                ? as text, 
                ?::DOUBLE[] AS embedding
        ),

        q2 AS (
            select 
                distinct text, 
                embedding::DOUBLE[] as embedding
            from embeddings
        )

        SELECT 
            b.text, 
            list_cosine_similarity(a.embedding::DOUBLE[], b.embedding::DOUBLE[]) AS similarity
        FROM q1 a
        join q2 b on a.text != b.text
        ORDER BY similarity DESC
        LIMIT 10
        """

    embedding = duckdb_embeddings([text], model, con)[0]
    result = con.execute(sql, [text, embedding]).fetchall()
    return result
