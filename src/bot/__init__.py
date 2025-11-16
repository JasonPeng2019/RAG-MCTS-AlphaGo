"""
DataGo Bot Package

A RAG-enhanced Go bot that integrates KataGo's MCTS with a retrieval-augmented
generation system for improved play on uncertain/complex positions.
"""

from .datago_bot import DataGoBot
from .gomill_player import GomillPlayer

__all__ = ["DataGoBot", "GomillPlayer"]
