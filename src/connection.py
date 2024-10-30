import duckdb
from duckdb import DuckDBPyConnection

def open_connection(dbname=None) -> duckdb.DuckDBPyConnection:
    """
    Connects to a local duckdb database and returns a connection object.

    Args:
        dbname (str, optional): The name of the database to connect to. If not provided, a connection to an in-memory database will be established.

    Returns:
        `duckdb.DuckDBPyConnection`: A connection object to the local database.
    """
    try:
        if dbname:
            return duckdb.connect(f"{dbname}.db")
        else:
            return duckdb.connect(":memory:")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        raise

def load_extension(
    con: duckdb.DuckDBPyConnection, extension: str
) -> DuckDBPyConnection:
    """
    Loads and installs a DuckDB extension in the given connection.

    Args:
        con (`duckdb.DuckDBPyConnection`): The DuckDB connection.
        extension (str): The name of the extension to load.

    Returns:
        `duckdb.DuckDBPyConnection`: The modified DuckDB connection.

    Raises:
        Any exception raised during the installation or loading of the extension.
    """
    try:
        con.install_extension(extension)
        con.load_extension(extension)
    except Exception as e:
        print(f"Could not load extension {extension}: {e}")
        raise
    return con
