from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from weaviate_client import WeaviateClientManager

load_dotenv()


@dataclass
class RetrievedDocument:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class WeaviateHybridRetriever:
    def __init__(
        self,
        weaviate_mode: str = "local",
        class_name: str = "MRInsights",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        alpha: float = 0.5,
        top_k: int = 5,
    ):
        self.client = WeaviateClientManager(mode=weaviate_mode).get_client()
        self.class_name = class_name
        self.model = SentenceTransformer(embedding_model_name)
        self.alpha = alpha
        self.top_k = top_k

    def _build_where_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict]:
        if not filters:
            return None

        operands = []
        for key, value in filters.items():
            if value is None:
                continue

            if isinstance(value, str):
                operands.append(
                    {
                        "path": [key],
                        "operator": "Equal",
                        "valueText": value,
                    }
                )
            elif isinstance(value, bool):
                operands.append(
                    {
                        "path": [key],
                        "operator": "Equal",
                        "valueBoolean": value,
                    }
                )
            elif isinstance(value, int):
                operands.append(
                    {
                        "path": [key],
                        "operator": "Equal",
                        "valueInt": value,
                    }
                )
            elif isinstance(value, float):
                operands.append(
                    {
                        "path": [key],
                        "operator": "Equal",
                        "valueNumber": value,
                    }
                )

        if not operands:
            return None

        if len(operands) == 1:
            return operands[0]

        return {"operator": "And", "operands": operands}

    def _embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[RetrievedDocument]:
        top_k = top_k or self.top_k
        alpha = self.alpha if alpha is None else alpha

        where_filter = self._build_where_filter(filters)
        query_vector = self._embed_query(query)

        properties = [
            "content",
            "doctor_id",
            "doctor_name",
            "specialty",
            "region",
            "date",
            "product",
            "source",
            "file_name",
        ]

        query_builder = (
            self.client.query.get(self.class_name, properties)
            .with_hybrid(query=query, alpha=alpha, vector=query_vector)
            .with_limit(top_k)
        )

        if where_filter:
            query_builder = query_builder.with_where(where_filter)

        result = query_builder.do()
        raw_hits = result.get("data", {}).get("Get", {}).get(self.class_name, [])

        docs: List[RetrievedDocument] = []
        for item in raw_hits:
            docs.append(
                RetrievedDocument(
                    page_content=item.get("content", ""),
                    metadata={
                        "doctor_id": item.get("doctor_id"),
                        "doctor_name": item.get("doctor_name"),
                        "specialty": item.get("specialty"),
                        "region": item.get("region"),
                        "date": item.get("date"),
                        "product": item.get("product"),
                        "source": item.get("source"),
                        "file_name": item.get("file_name"),
                    },
                )
            )

        return docs

    def print_results(self, docs: List[RetrievedDocument]) -> None:
        if not docs:
            print("No results found.")
            return

        for i, doc in enumerate(docs, start=1):
            print(f"\n--- Result {i} ---")
            print(f"Doctor Name : {doc.metadata.get('doctor_name')}")
            print(f"Specialty   : {doc.metadata.get('specialty')}")
            print(f"Region      : {doc.metadata.get('region')}")
            print(f"Date        : {doc.metadata.get('date')}")
            print(f"Product     : {doc.metadata.get('product')}")
            print(f"Source      : {doc.metadata.get('source')}")
            print(f"File Name   : {doc.metadata.get('file_name')}")
            print(f"Content     : {doc.page_content}")


if __name__ == "__main__":
    retriever = WeaviateHybridRetriever(
        weaviate_mode="cloud",   # or "local"
        class_name="MRInsights",
        alpha=0.5,
        top_k=5,
    )

    query = "common objections about cost and long term safety"
    filters = {
        "specialty": "Endocrinologist",
        "region": "India",
    }

    results = retriever.search(query=query, filters=filters)
    retriever.print_results(results)