"""
Definicje zadań uczenia dla agentów
"""

import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass


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


class TaskFactory:
    """Fabryka zadań uczenia"""
    
    @staticmethod
    def get_task(task_name: str) -> Task:
        """Zwraca zadanie po nazwie"""
        tasks = {
            'xor': TaskFactory.create_xor(),
            'and': TaskFactory.create_and(),
            'or': TaskFactory.create_or(),
            'nand': TaskFactory.create_nand(),
            'nor': TaskFactory.create_nor(),
            'xnor': TaskFactory.create_xnor(),
            '3bit_parity': TaskFactory.create_3bit_parity(),
            'majority': TaskFactory.create_majority(),
        }
        
        if task_name not in tasks:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(tasks.keys())}")
        
        return tasks[task_name]
    
    @staticmethod
    def list_tasks() -> Dict[str, Dict[str, Any]]:
        """Lista wszystkich dostępnych zadań"""
        return {
            'xor': {'difficulty': 'medium', 'description': 'Exclusive OR - klasyczny problem nieliniowy'},
            'and': {'difficulty': 'easy', 'description': 'Logical AND - prosty problem liniowy'},
            'or': {'difficulty': 'easy', 'description': 'Logical OR - prosty problem liniowy'},
            'nand': {'difficulty': 'easy', 'description': 'NOT AND - prosty problem liniowy'},
            'nor': {'difficulty': 'easy', 'description': 'NOT OR - prosty problem liniowy'},
            'xnor': {'difficulty': 'medium', 'description': 'NOT XOR - problem nieliniowy'},
            '3bit_parity': {'difficulty': 'hard', 'description': '3-bit parity check - trudny problem nieliniowy'},
            'majority': {'difficulty': 'medium', 'description': '3-bit majority vote - problem nieliniowy'},
        }
    
    @staticmethod
    def create_xor() -> Task:
        """XOR: 0⊕0=0, 0⊕1=1, 1⊕0=1, 1⊕1=0"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        return Task(
            name='XOR',
            X=X,
            y=y,
            input_dim=2,
            output_dim=1,
            description='Exclusive OR - klasyczny problem nieliniowy',
            difficulty='medium'
        )
    
    @staticmethod
    def create_and() -> Task:
        """AND: 0∧0=0, 0∧1=0, 1∧0=0, 1∧1=1"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])
        
        return Task(
            name='AND',
            X=X,
            y=y,
            input_dim=2,
            output_dim=1,
            description='Logical AND - prosty problem liniowy',
            difficulty='easy'
        )
    
    @staticmethod
    def create_or() -> Task:
        """OR: 0∨0=0, 0∨1=1, 1∨0=1, 1∨1=1"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [1]])
        
        return Task(
            name='OR',
            X=X,
            y=y,
            input_dim=2,
            output_dim=1,
            description='Logical OR - prosty problem liniowy',
            difficulty='easy'
        )
    
    @staticmethod
    def create_nand() -> Task:
        """NAND: NOT(AND) - 0∧0=1, 0∧1=1, 1∧0=1, 1∧1=0"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[1], [1], [1], [0]])
        
        return Task(
            name='NAND',
            X=X,
            y=y,
            input_dim=2,
            output_dim=1,
            description='NOT AND - prosty problem liniowy',
            difficulty='easy'
        )
    
    @staticmethod
    def create_nor() -> Task:
        """NOR: NOT(OR) - 0∨0=1, 0∨1=0, 1∨0=0, 1∨1=0"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[1], [0], [0], [0]])
        
        return Task(
            name='NOR',
            X=X,
            y=y,
            input_dim=2,
            output_dim=1,
            description='NOT OR - prosty problem liniowy',
            difficulty='easy'
        )
    
    @staticmethod
    def create_xnor() -> Task:
        """XNOR: NOT(XOR) - 0⊕0=1, 0⊕1=0, 1⊕0=0, 1⊕1=1"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[1], [0], [0], [1]])
        
        return Task(
            name='XNOR',
            X=X,
            y=y,
            input_dim=2,
            output_dim=1,
            description='NOT XOR - problem nieliniowy',
            difficulty='medium'
        )
    
    @staticmethod
    def create_3bit_parity() -> Task:
        """
        3-bit parity: zwraca 1 jeśli nieparzysta liczba bitów = 1
        Trudniejszy problem - wymaga większej sieci
        """
        X = np.array([
            [0, 0, 0],  # 0 jedynek -> 0
            [0, 0, 1],  # 1 jedynka -> 1
            [0, 1, 0],  # 1 jedynka -> 1
            [0, 1, 1],  # 2 jedynki -> 0
            [1, 0, 0],  # 1 jedynka -> 1
            [1, 0, 1],  # 2 jedynki -> 0
            [1, 1, 0],  # 2 jedynki -> 0
            [1, 1, 1],  # 3 jedynki -> 1
        ])
        y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])
        
        return Task(
            name='3-bit Parity',
            X=X,
            y=y,
            input_dim=3,
            output_dim=1,
            description='3-bit parity check - trudny problem nieliniowy',
            difficulty='hard'
        )
    
    @staticmethod
    def create_majority() -> Task:
        """
        3-bit majority: zwraca 1 jeśli większość bitów = 1
        """
        X = np.array([
            [0, 0, 0],  # 0 jedynek -> 0
            [0, 0, 1],  # 1 jedynka -> 0
            [0, 1, 0],  # 1 jedynka -> 0
            [0, 1, 1],  # 2 jedynki -> 1
            [1, 0, 0],  # 1 jedynka -> 0
            [1, 0, 1],  # 2 jedynki -> 1
            [1, 1, 0],  # 2 jedynki -> 1
            [1, 1, 1],  # 3 jedynki -> 1
        ])
        y = np.array([[0], [0], [0], [1], [0], [1], [1], [1]])
        
        return Task(
            name='Majority Vote',
            X=X,
            y=y,
            input_dim=3,
            output_dim=1,
            description='3-bit majority vote - problem nieliniowy',
            difficulty='medium'
        )


if __name__ == "__main__":
    print("🎯 Dostępne zadania uczenia:\n")
    
    tasks_info = TaskFactory.list_tasks()
    for task_name, info in tasks_info.items():
        task = TaskFactory.get_task(task_name)
        print(f"📌 {task.name}")
        print(f"   Trudność: {info['difficulty']}")
        print(f"   Opis: {info['description']}")
        print(f"   Wymiary: {task.input_dim} → {task.output_dim}")
        print(f"   Przykłady: {len(task.X)}")
        print()
    
    # Test XOR
    print("\n🧪 Test zadania XOR:")
    xor_task = TaskFactory.get_task('xor')
    print(f"Wejścia:\n{xor_task.X}")
    print(f"Oczekiwane wyjścia:\n{xor_task.y}")
    
    # Test 3-bit parity
    print("\n🧪 Test zadania 3-bit Parity:")
    parity_task = TaskFactory.get_task('3bit_parity')
    print(f"Wejścia:\n{parity_task.X}")
    print(f"Oczekiwane wyjścia:\n{parity_task.y}")
