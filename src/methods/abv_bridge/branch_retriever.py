"""Branch Retriever: runs bridge-conditioned retrieval for each proposal.

For each bridge proposal, constructs a focused query:
    original_question + " " + bridge_entity + " " + bridge_relation

Then retrieves top-k chunks using the existing TextRetriever.
"""

from typing import Dict, List, Any
from src.tools.text_tools import TextRetriever


class BranchRetriever:
    """Retrieves evidence conditioned on each bridge proposal."""

    def __init__(self, text_retriever: TextRetriever, top_k: int = 10):
        self.text_retriever = text_retriever
        self.top_k = top_k

    def retrieve_branches(self, question: str, question_id: str,
                          proposals: List[Dict]) -> List[Dict[str, Any]]:
        """Retrieve for each proposal. Returns list of branch results.

        Each branch result:
        {
            "proposal": Dict,           # The bridge proposal
            "chunks": List[Dict],       # Retrieved chunks
            "chunk_ids": set,           # For overlap computation
            "query": str,              # The augmented query used
        }
        """
        branches = []
        for prop in proposals:
            entity = prop.get("bridge_entity", "")
            relation = prop.get("bridge_relation", "")
            # Build focused query: question + bridge entity + relation hint
            parts = [question]
            if entity:
                parts.append(entity)
            if relation and len(relation) < 40:
                parts.append(relation)
            aug_query = " ".join(parts)

            chunks = self.text_retriever.retrieve(
                aug_query, top_k=self.top_k, question_id=question_id
            )
            branches.append({
                "proposal": prop,
                "chunks": chunks,
                "chunk_ids": {c["chunk_id"] for c in chunks},
                "query": aug_query,
            })

        return branches
