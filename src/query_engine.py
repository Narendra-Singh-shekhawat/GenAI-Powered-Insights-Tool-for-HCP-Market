import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from groq import Groq

from sales_query_service import SalesQueryService
from retriever import WeaviateHybridRetriever, RetrievedDocument


load_dotenv()


class QueryEngine:
    def __init__(self):
        self.sales_service = SalesQueryService()
        self.retriever = WeaviateHybridRetriever(
            weaviate_mode="cloud",   # change to "local" if needed
            class_name="MRInsights",
            alpha=0.5,
            top_k=8,
        )

        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY in .env")

        self.groq_model = groq_model
        self.llm_client = Groq(api_key=groq_api_key)

    # --------------------------------------------------
    # Structured step: get HCPs with prescription drop
    # --------------------------------------------------
    def get_declining_hcps(self, specialty: str, product: str) -> List[Dict[str, Any]]:
        rows = self.sales_service.get_hcps_with_prescription_drop(
            specialty=specialty,
            product=product
        )
        return rows

    # --------------------------------------------------
    # Unstructured step: retrieve MR docs for given HCPs
    # --------------------------------------------------
    def get_mr_docs_for_hcps(
        self,
        doctor_ids: List[str],
        query: str,
        product: str = None,
        region: str = None,
        top_k_per_hcp: int = 3
    ) -> List[RetrievedDocument]:
        all_docs: List[RetrievedDocument] = []

        for doctor_id in doctor_ids:
            filters = {"doctor_id": doctor_id}

            if product:
                filters["product"] = product
            if region:
                filters["region"] = region

            docs = self.retriever.search(
                query=query,
                filters=filters,
                top_k=top_k_per_hcp,
                alpha=0.6
            )
            all_docs.extend(docs)

        return all_docs

    # --------------------------------------------------
    # Build context for LLM
    # --------------------------------------------------
    def build_context(
        self,
        declining_hcps: List[Dict[str, Any]],
        mr_docs: List[RetrievedDocument]
    ) -> str:
        sales_lines = []
        for row in declining_hcps:
            sales_lines.append(
                f"Doctor ID: {row.get('doctor_id')}, "
                f"Doctor Name: {row.get('doctor_name')}, "
                f"Specialty: {row.get('specialty')}, "
                f"Region: {row.get('region')}, "
                f"Product: {row.get('product')}, "
                f"Quarter: {row.get('quarter')}, "
                f"Current Prescriptions: {row.get('prescriptions')}, "
                f"Previous Prescriptions: {row.get('prev_prescriptions')}, "
                f"Drop Value: {row.get('drop_value')}"
            )

        mr_lines = []
        for i, doc in enumerate(mr_docs, start=1):
            mr_lines.append(
                f"[MR Evidence {i}] "
                f"Doctor ID: {doc.metadata.get('doctor_id')}, "
                f"Doctor Name: {doc.metadata.get('doctor_name')}, "
                f"Specialty: {doc.metadata.get('specialty')}, "
                f"Region: {doc.metadata.get('region')}, "
                f"Product: {doc.metadata.get('product')}, "
                f"Source: {doc.metadata.get('source')}, "
                f"Content: {doc.page_content}"
            )

        context = (
            "STRUCTURED SALES EVIDENCE:\n"
            + ("\n".join(sales_lines) if sales_lines else "No structured sales evidence found.")
            + "\n\n"
            + "UNSTRUCTURED MR EVIDENCE:\n"
            + ("\n".join(mr_lines) if mr_lines else "No MR evidence found.")
        )
        return context

    # --------------------------------------------------
    # LLM synthesis
    # --------------------------------------------------
    def synthesize_answer(self, user_query: str, context: str) -> str:
        system_prompt = """
You are a pharma commercial insights assistant.

Your job:
- Answer only from the provided evidence
- Do not invent facts
- Identify common objection themes from MR evidence
- Tie them back to the declining HCP cohort from structured sales evidence
- Be concise but clear
- Include:
  1. Summary
  2. Top objections/themes
  3. Supporting evidence snippets
  4. Cohort notes / limitations

If evidence is weak or incomplete, say so explicitly.
"""

        user_prompt = f"""
User question:
{user_query}

Evidence:
{context}

Please generate a grounded answer.
"""

        response = self.llm_client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content

    # --------------------------------------------------
    # End-to-end mixed query
    # --------------------------------------------------
    def answer_decline_plus_objections_query(
        self,
        user_query: str,
        specialty: str = "Endocrinologist",
        product: str = "GlucoX",
        region: str = "India"
    ) -> Dict[str, Any]:
        declining_hcps = self.get_declining_hcps(
            specialty=specialty,
            product=product
        )

        if not declining_hcps:
            return {
                "answer": "No declining HCPs found for the selected specialty/product.",
                "declining_hcps": [],
                "mr_docs_count": 0
            }

        doctor_ids = list({row["doctor_id"] for row in declining_hcps})

        mr_docs = self.get_mr_docs_for_hcps(
            doctor_ids=doctor_ids,
            query="objections barriers concerns unmet needs prescribing hesitation",
            product=product,
            region=region,
            top_k_per_hcp=3
        )

        context = self.build_context(declining_hcps, mr_docs)
        answer = self.synthesize_answer(user_query, context)

        return {
            "answer": answer,
            "declining_hcps": declining_hcps,
            "mr_docs_count": len(mr_docs),
            "doctor_ids": doctor_ids
        }


if __name__ == "__main__":
    engine = QueryEngine()

    query = (
        "What are the most common objections raised by endocrinologists "
        "whose prescription volume dropped in the past 2 quarters?"
    )

    result = engine.answer_decline_plus_objections_query(
        user_query=query,
        specialty="Endocrinologist",
        product="GlucoX",
        region="India"
    )

    print("\n========== FINAL ANSWER ==========\n")
    print(result["answer"])

    print("\n========== DEBUG INFO ==========\n")
    print(f"Declining HCP count: {len(result['declining_hcps'])}")
    print(f"MR docs retrieved: {result['mr_docs_count']}")
    print(f"Doctor IDs: {result['doctor_ids']}")