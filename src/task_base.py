"""
Task dataclass - definicja bazowa dla wszystkich zadań
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple, Dict, Any


@dataclass
class Task:
    """Definicja zadania uczenia"""
    name: str
    X: np.ndarray  # Dane wejściowe
    y: np.ndarray  # Oczekiwane wyjścia
    input_dim: int
    output_dim: int
    description: str
    difficulty: str  # 'easy', 'medium', 'hard'
