"""
Definicje zadań uczenia dla agentów
"""

import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import extended_tasks


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
            # Logiczne - podstawowe
            'xor': TaskFactory.create_xor(),
            'and': TaskFactory.create_and(),
            'or': TaskFactory.create_or(),
            'nand': TaskFactory.create_nand(),
            'nor': TaskFactory.create_nor(),
            'xnor': TaskFactory.create_xnor(),
            
            # Logiczne - zaawansowane
            '3bit_parity': TaskFactory.create_3bit_parity(),
            '4bit_parity': TaskFactory.create_4bit_parity(),
            'majority': TaskFactory.create_majority(),
            'minority': TaskFactory.create_minority(),
            'implication': TaskFactory.create_implication(),
            'equivalence': TaskFactory.create_equivalence(),
            
            # Arytmetyczne
            'addition_mod2': TaskFactory.create_addition_mod2(),
            'subtraction_mod2': TaskFactory.create_subtraction_mod2(),
            'multiplication_mod2': TaskFactory.create_multiplication_mod2(),
            'is_even': TaskFactory.create_is_even(),
            'is_power_of_two': TaskFactory.create_is_power_of_two(),
            'greater_than': TaskFactory.create_greater_than(),
            
            # Pattern recognition
            'alternating_pattern': TaskFactory.create_alternating_pattern(),
            'repeating_pattern': TaskFactory.create_repeating_pattern(),
            'symmetric_pattern': TaskFactory.create_symmetric_pattern(),
            'palindrome': TaskFactory.create_palindrome(),
            'monotonic_sequence': TaskFactory.create_monotonic_sequence(),
            
            # Sequence learning
            'fibonacci_mod2': TaskFactory.create_fibonacci_mod2(),
            'prime_detection': TaskFactory.create_prime_detection(),
            'count_ones': TaskFactory.create_count_ones(),
            'hamming_weight': TaskFactory.create_hamming_weight(),
            'gray_code': TaskFactory.create_gray_code(),
            
            # Complex logic
            'boolean_expression': extended_tasks.ExtendedTaskFactory.create_boolean_expression(),
            'circuit_simulation': extended_tasks.ExtendedTaskFactory.create_circuit_simulation(),
            'state_machine': extended_tasks.ExtendedTaskFactory.create_state_machine(),
            'binary_decoder': extended_tasks.ExtendedTaskFactory.create_binary_decoder(),
            
            # Dodaj wszystkie nowe zadania z ExtendedTaskFactory
            '4bit_parity': extended_tasks.ExtendedTaskFactory.create_4bit_parity(),
            'minority': extended_tasks.ExtendedTaskFactory.create_minority(),
            'implication': extended_tasks.ExtendedTaskFactory.create_implication(),
            'equivalence': extended_tasks.ExtendedTaskFactory.create_equivalence(),
            'addition_mod2': extended_tasks.ExtendedTaskFactory.create_addition_mod2(),
            'subtraction_mod2': extended_tasks.ExtendedTaskFactory.create_subtraction_mod2(),
            'multiplication_mod2': extended_tasks.ExtendedTaskFactory.create_multiplication_mod2(),
            'is_even': extended_tasks.ExtendedTaskFactory.create_is_even(),
            'is_power_of_two': extended_tasks.ExtendedTaskFactory.create_is_power_of_two(),
            'greater_than': extended_tasks.ExtendedTaskFactory.create_greater_than(),
            'alternating_pattern': extended_tasks.ExtendedTaskFactory.create_alternating_pattern(),
            'repeating_pattern': extended_tasks.ExtendedTaskFactory.create_repeating_pattern(),
            'symmetric_pattern': extended_tasks.ExtendedTaskFactory.create_symmetric_pattern(),
            'palindrome': extended_tasks.ExtendedTaskFactory.create_palindrome(),
            'monotonic_sequence': extended_tasks.ExtendedTaskFactory.create_monotonic_sequence(),
            'fibonacci_mod2': extended_tasks.ExtendedTaskFactory.create_fibonacci_mod2(),
            'prime_detection': extended_tasks.ExtendedTaskFactory.create_prime_detection(),
            'count_ones': extended_tasks.ExtendedTaskFactory.create_count_ones(),
            'hamming_weight': extended_tasks.ExtendedTaskFactory.create_hamming_weight(),
            'gray_code': extended_tasks.ExtendedTaskFactory.create_gray_code(),
        }
        
        if task_name not in tasks:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(tasks.keys())}")
        
        return tasks[task_name]
    
    @staticmethod
    def list_tasks() -> Dict[str, Dict[str, Any]]:
        """Lista wszystkich dostępnych zadań"""
        return {
            # Logiczne - podstawowe
            'xor': {'difficulty': 'medium', 'description': 'Exclusive OR - klasyczny problem nieliniowy'},
            'and': {'difficulty': 'easy', 'description': 'Logical AND - prosty problem liniowy'},
            'or': {'difficulty': 'easy', 'description': 'Logical OR - prosty problem liniowy'},
            'nand': {'difficulty': 'easy', 'description': 'Logical NAND - negacja koniunkcji'},
            'nor': {'difficulty': 'easy', 'description': 'Logical NOR - negacja alternatywy'},
            'xnor': {'difficulty': 'medium', 'description': 'Logical XNOR - równoważność'},
            
            # Logiczne - zaawansowane
            '3bit_parity': {'difficulty': 'medium', 'description': 'Parity 3-bit - sprawdzanie parzystości'},
            '4bit_parity': {'difficulty': 'hard', 'description': 'Parity 4-bit - trudne parzystości'},
            'majority': {'difficulty': 'medium', 'description': 'Majority - głosowanie większościowe'},
            'minority': {'difficulty': 'medium', 'description': 'Minority - głosowanie mniejszościowe'},
            'implication': {'difficulty': 'medium', 'description': 'Implication - implikacja logiczna'},
            'equivalence': {'difficulty': 'medium', 'description': 'Equivalence - równoważność logiczna'},
            
            # Arytmetyczne
            'addition_mod2': {'difficulty': 'medium', 'description': 'Addition mod 2 - dodawanie modulo 2'},
            'subtraction_mod2': {'difficulty': 'medium', 'description': 'Subtraction mod 2 - odejmowanie modulo 2'},
            'multiplication_mod2': {'difficulty': 'medium', 'description': 'Multiplication mod 2 - mnożenie modulo 2'},
            'is_even': {'difficulty': 'easy', 'description': 'Is even - sprawdzanie parzystości liczby'},
            'is_power_of_two': {'difficulty': 'hard', 'description': 'Is power of two - sprawdzanie potęgi dwójki'},
            'greater_than': {'difficulty': 'medium', 'description': 'Greater than - porównywanie liczb'},
            
            # Pattern recognition
            'alternating_pattern': {'difficulty': 'medium', 'description': 'Alternating pattern - rozpoznawanie naprzemianności'},
            'repeating_pattern': {'difficulty': 'medium', 'description': 'Repeating pattern - rozpoznawanie powtórzeń'},
            'symmetric_pattern': {'difficulty': 'medium', 'description': 'Symmetric pattern - rozpoznawanie symetrii'},
            'palindrome': {'difficulty': 'hard', 'description': 'Palindrome - sprawdzanie palindromów'},
            'monotonic_sequence': {'difficulty': 'medium', 'description': 'Monotonic sequence - rozpoznawanie monotoniczności'},
            
            # Sequence learning
            'fibonacci_mod2': {'difficulty': 'hard', 'description': 'Fibonacci mod 2 - ciąg Fibonacciego modulo 2'},
            'prime_detection': {'difficulty': 'hard', 'description': 'Prime detection - wykrywanie liczb pierwszych'},
            'count_ones': {'difficulty': 'medium', 'description': 'Count ones - zliczanie jedynek binarnych'},
            'hamming_weight': {'difficulty': 'medium', 'description': 'Hamming weight - waga Hamminga'},
            'gray_code': {'difficulty': 'hard', 'description': 'Gray code - kod Graya'},
            
            # Complex logic
            'boolean_expression': {'difficulty': 'hard', 'description': 'Boolean expression - złożone wyrażenia logiczne'},
            'circuit_simulation': {'difficulty': 'hard', 'description': 'Circuit simulation - symulacja układów logicznych'},
            'state_machine': {'difficulty': 'hard', 'description': 'State machine - maszyna stanów'},
            'binary_decoder': {'difficulty': 'medium', 'description': 'Binary decoder - dekoder binarny'},
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
