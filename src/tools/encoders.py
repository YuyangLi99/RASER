"""Unified encoder wrappers for dense retrieval.

Two backends:
1. NomicEncoder: nomic-ai/nomic-embed-text-v1.5 (137M, 768-d, sentence-transformers)
2. GteQwenEncoder: Alibaba-NLP/gte-Qwen2-7B-instruct (7B, 3584-d, raw transformers + last-token pool)

Both expose the same interface:
    encode_queries(texts: List[str]) -> np.ndarray  (n, d), L2-normalized
    encode_documents(texts: List[str]) -> np.ndarray  (n, d), L2-normalized
"""

from typing import List
import numpy as np
import torch
import torch.nn.functional as F


class NomicEncoder:
    name = "nomic-v1.5"
    dim = 768

    def __init__(self, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            device=device,
        )
        self.model.eval()

    @torch.no_grad()
    def encode_queries(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        prefixed = [f"search_query: {t}" for t in texts]
        return self.model.encode(prefixed, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

    @torch.no_grad()
    def encode_documents(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        prefixed = [f"search_document: {t}" for t in texts]
        return self.model.encode(prefixed, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)


class GteQwenEncoder:
    name = "gte-qwen2-7b"
    dim = 3584

    def __init__(self, device: str = "cuda", max_length: int = 512):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(device).eval()
        self.device = device
        self.max_length = max_length

    @staticmethod
    def _last_token_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        left_padding = (mask[:, -1].sum() == mask.shape[0])
        if left_padding:
            return h[:, -1]
        seq_len = mask.sum(dim=1) - 1
        return h[torch.arange(h.shape[0], device=h.device), seq_len]

    @torch.no_grad()
    def _encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[i : i + batch_size],
                padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            h = self.model(**batch).last_hidden_state
            emb = self._last_token_pool(h, batch["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def encode_queries(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        task = "Given a multi-hop question, retrieve passages and triples that contain evidence to answer it"
        prefixed = [f"Instruct: {task}\nQuery: {t}" for t in texts]
        return self._encode(prefixed, batch_size)

    def encode_documents(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        return self._encode(texts, batch_size)


def get_encoder(name: str, device: str = "cuda"):
    name = name.lower()
    if name in ("nomic", "nomic-v1.5"):
        return NomicEncoder(device=device)
    if name in ("gte-qwen", "gte-qwen2", "gte-qwen2-7b"):
        return GteQwenEncoder(device=device)
    raise ValueError(f"Unknown encoder: {name}")
