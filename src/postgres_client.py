import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()


class PostgresClient:
    def __init__(self):
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = os.getenv("POSTGRES_PORT", "5432")
        self.dbname = os.getenv("POSTGRES_DB", "pharma_case_study")
        self.user = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD", "postgres")

    def get_connection(self):
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
        )

    def test_connection(self):
        conn = None
        try:
            conn = self.get_connection()
            print("✅ PostgreSQL connection successful")
        except Exception as e:
            print(f"❌ PostgreSQL connection failed: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def execute_query(self, query, params=None, fetch=True):
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)

            if fetch:
                results = cursor.fetchall()
                return [dict(row) for row in results]

            conn.commit()
            return None

        except Exception as e:
            if conn:
                conn.rollback()
            print(f"❌ Query execution failed: {e}")
            raise

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()


if __name__ == "__main__":
    client = PostgresClient()
    client.test_connection()