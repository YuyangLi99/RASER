"""LLM-based evidence verifier using KIT API."""

import json
from openai import OpenAI
from typing import Dict, List, Any

VERIFIER_SYSTEM_PROMPT = """You are an evidence verifier for a multi-hop question answering system.

Given a question, a candidate answer, and supporting evidence (both graph-based and text-based), you must evaluate:

1. Is the candidate answer supported by the graph evidence?
2. Is the candidate answer supported by the text evidence?
3. Do the graph and text evidence conflict with each other?
4. Overall, is the evidence sufficient to confidently answer?

Return valid JSON only:
{
  "graph_support": 0.0 to 1.0,
  "text_support": 0.0 to 1.0,
  "conflict_score": 0.0 to 1.0,
  "sufficiency_score": 0.0 to 1.0,
  "verdict": "supported" | "weakly_supported" | "unsupported" | "conflicting",
  "reasoning": "brief explanation"
}"""


class LLMVerifier:
    """Verifies candidate answers against evidence using LLM."""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-oss-120b"):
        self.api_key = api_key or ""
        self.base_url = base_url or ""
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.total_tokens_used = 0

    def verify(self, question: str, candidate_answer: str,
               graph_evidence: List[Dict], text_evidence: List[Dict]) -> Dict[str, Any]:
        """Verify a candidate answer against graph and text evidence."""
        user_msg = self._format_verification(question, candidate_answer, graph_evidence, text_evidence)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                timeout=30,
            )
            content = response.choices[0].message.content.strip()
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens

            return self._parse_response(content)

        except Exception as e:
            print(f"[Verifier Error] {e}")
            return {
                "graph_support": 0.0,
                "text_support": 0.0,
                "conflict_score": 0.5,
                "sufficiency_score": 0.0,
                "verdict": "unsupported",
                "reasoning": f"Verifier error: {str(e)[:100]}",
            }

    def _format_verification(self, question: str, candidate_answer: str,
                              graph_evidence: List[Dict], text_evidence: List[Dict]) -> str:
        parts = [
            f"## Question\n{question}",
            f"## Candidate Answer\n{candidate_answer}",
        ]

        if graph_evidence:
            ge = "\n".join(f"- {e.get('content', e.get('linearized', str(e)))[:200]}" for e in graph_evidence[:5])
            parts.append(f"## Graph Evidence\n{ge}")
        else:
            parts.append("## Graph Evidence\nNone.")

        if text_evidence:
            te = "\n".join(f"- {e.get('content', e.get('text', str(e)))[:200]}" for e in text_evidence[:5])
            parts.append(f"## Text Evidence\n{te}")
        else:
            parts.append("## Text Evidence\nNone.")

        parts.append("\nEvaluate the evidence and return JSON only.")
        return "\n\n".join(parts)

    def _parse_response(self, content: str) -> dict:
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(content)
            # Ensure required fields with defaults
            return {
                "graph_support": float(parsed.get("graph_support", 0)),
                "text_support": float(parsed.get("text_support", 0)),
                "conflict_score": float(parsed.get("conflict_score", 0)),
                "sufficiency_score": float(parsed.get("sufficiency_score", 0)),
                "verdict": parsed.get("verdict", "unsupported"),
                "reasoning": parsed.get("reasoning", ""),
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "graph_support": 0.0,
                "text_support": 0.0,
                "conflict_score": 0.5,
                "sufficiency_score": 0.0,
                "verdict": "unsupported",
                "reasoning": "Could not parse verifier response",
            }
