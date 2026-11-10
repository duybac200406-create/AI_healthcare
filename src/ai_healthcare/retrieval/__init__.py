"""Retrieval services for the AI Healthcare application."""

from ai_healthcare.retrieval.dense_retriever import (
    DenseRetriever,
    DenseRetrievalError,
)

__all__ = ["DenseRetriever", "DenseRetrievalError"]
