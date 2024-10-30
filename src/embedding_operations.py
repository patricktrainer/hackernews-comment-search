from typing import List, Tuple
from .connection import DuckDBPyConnection
from .operations import EmbeddingKey, load_pickle_cache, save_pickle_cache, is_key_in_table, get_embedding_from_table, write_embedding_to_table
import src.openai_client as openai_client

class EmbeddingOperations:
    def __init__(self, con: DuckDBPyConnection, pickle_path: str):
        self.con = con
        self.pickle_path = pickle_path

    def pickle_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        embeddings = []
        pickle_cache = load_pickle_cache(self.pickle_path)

        for text in texts:
            key = EmbeddingKey(text, model)
            if key not in pickle_cache:
                pickle_cache[key] = openai_client.create_embedding(text, model=model)
            embeddings.append(pickle_cache[key])
        save_pickle_cache(pickle_cache, self.pickle_path)
        return embeddings

    def duckdb_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        embeddings = []
        for text in texts:
            key = EmbeddingKey(text, model)
            result = is_key_in_table(self.con, key)
            if result:
                embedding = get_embedding_from_table(self.con, key)
                embeddings.append(embedding)
            else:
                embedding = openai_client.create_embedding(text, model)
                write_embedding_to_table(self.con, key, embedding)
                embeddings.append(embedding)
        return embeddings

    def cosine_similarity(self, l1, l2) -> float:
        return self.con.execute(f"SELECT list_cosine_similarity({l1}, {l2})").fetchall()[0][0]

    def get_similarity(self, text: str, model: str) -> list[tuple[str, float]]:
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

        embedding = self.duckdb_embeddings([text], model)[0]
        result = self.con.execute(sql, [text, embedding]).fetchall()
        return result
