import os
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey


# Load environment variables from .env (if present)
load_dotenv()


class WeaviateClientManager:
    def __init__(self, mode="local"):
        """
        mode = "local"  → Docker Weaviate
        mode = "cloud"  → Weaviate Cloud (API key)
        """
        self.mode = mode
        self.client = self._create_client()

    def _create_client(self):
        if self.mode == "local":
            return self._local_client()
        elif self.mode == "cloud":
            return self._cloud_client()
        else:
            raise ValueError("Invalid mode. Use 'local' or 'cloud'")

    def _local_client(self):
        url = os.getenv("WEAVIATE_LOCAL_URL", "http://localhost:8080")

        client = weaviate.Client(url)

        if not client.is_ready():
            raise ConnectionError("❌ Local Weaviate is not running")

        print("✅ Connected to LOCAL Weaviate")
        return client

    def _cloud_client(self):
        url = os.getenv("WEAVIATE_URL")
        api_key = os.getenv("WEAVIATE_API_KEY")

        if not url or not api_key:
            raise ValueError("❌ Missing WEAVIATE_URL or WEAVIATE_API_KEY")

        auth_config = AuthApiKey(api_key=api_key)

        client = weaviate.Client(
            url=url,
            auth_client_secret=auth_config,
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")
            }
        )

        if not client.is_ready():
            raise ConnectionError("❌ Cloud Weaviate not reachable")

        print("✅ Connected to CLOUD Weaviate")
        return client

    def get_client(self):
        return self.client