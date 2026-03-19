"""
Dodatkowe zadania uczenia dla agentów - rozszerzona kolekcja
"""

import numpy as np
from typing import Tuple, Dict, Any
from tasks import Task


class ExtendedTaskFactory:
    """Rozszerzona fabryka zadań uczenia"""
    
    @staticmethod
    def create_4bit_parity() -> Task:
        """4-bit parity: zwraca 1 jeśli nieparzysta liczba bitów = 1"""
        X = np.array([
            [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
            [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
            [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
            [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1],
        ])
        y = np.array([[0], [1], [1], [0], [1], [0], [0], [1], 
                     [1], [0], [0], [1], [0], [1], [1], [0]])
        
        return Task(
            name='4-bit Parity',
            X=X,
            y=y,
            input_dim=4,
            output_dim=1,
            description='4-bit parity check - bardzo trudny problem nieliniowy',
            difficulty='hard'
        )
    
    @staticmethod
    def create_minority() -> Task:
        """Minority: zwraca 1 jeśli mniejszość bitów = 1"""
        X = np.array([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ])
        y = np.array([[1], [1], [1], [0], [1], [0], [0], [0]])
        
        return Task(
            name='Minority',
            X=X,
            y=y,
            input_dim=3,
            output_dim=1,
            description='Minority function - głosowanie mniejszościowe',
            difficulty='medium'
        )
    
    @staticmethod
    def create_implication() -> Task:
        """Implication: A → B (jeśli A to B)"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[1], [1], [0], [1]])
        
        return Task(
            name='Implication',
            X=X,
            y=y,
            input_dim=2,
            output_dim=1,
            description='Logical implication - A → B',
            difficulty='medium'
        )
    
    @staticmethod
    def create_equivalence() -> Task:
        """Equivalence: A ↔ B (A wtedy i tylko wtedy gdy B)"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[1], [0], [0], [1]])
        
        return Task(
            name='Equivalence',
            X=X,
            y=y,
            input_dim=2,
            output_dim=1,
            description='Logical equivalence - A ↔ B',
            difficulty='medium'
        )
    
    @staticmethod
    def create_addition_mod2() -> Task:
        """Dodawanie modulo 2: (A + B) mod 2"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        return Task(
            name='Addition Mod 2',
            X=X,
            y=y,
            input_dim=2,
            output_dim=1,
            description='Binary addition modulo 2',
            difficulty='medium'
        )
    
    @staticmethod
    def create_subtraction_mod2() -> Task:
        """Odejmowanie modulo 2: (A - B) mod 2"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        return Task(
            name='Subtraction Mod 2',
            X=X,
            y=y,
            input_dim=2,
            output_dim=1,
            description='Binary subtraction modulo 2',
            difficulty='medium'
        )
    
    @staticmethod
    def create_multiplication_mod2() -> Task:
        """Mnożenie modulo 2: (A * B) mod 2"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])
        
        return Task(
            name='Multiplication Mod 2',
            X=X,
            y=y,
            input_dim=2,
            output_dim=1,
            description='Binary multiplication modulo 2',
            difficulty='medium'
        )
    
    @staticmethod
    def create_is_even() -> Task:
        """Sprawdzanie czy liczba jest parzysta"""
        X = np.array([[0], [1], [2], [3], [4], [5], [6], [7]])
        y = np.array([[1], [0], [1], [0], [1], [0], [1], [0]])
        
        return Task(
            name='Is Even',
            X=X,
            y=y,
            input_dim=1,
            output_dim=1,
            description='Check if number is even',
            difficulty='easy'
        )
    
    @staticmethod
    def create_is_power_of_two() -> Task:
        """Sprawdzanie czy liczba jest potęgą dwójki"""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
        y = np.array([[1], [1], [0], [1], [0], [0], [0], [1]])
        
        return Task(
            name='Is Power of Two',
            X=X,
            y=y,
            input_dim=1,
            output_dim=1,
            description='Check if number is power of two',
            difficulty='hard'
        )
    
    @staticmethod
    def create_greater_than() -> Task:
        """Porównywanie: A > B"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [2, 2]])
        y = np.array([[0], [0], [1], [0], [1], [1], [0]])
        
        return Task(
            name='Greater Than',
            X=X,
            y=y,
            input_dim=2,
            output_dim=1,
            description='Check if A > B',
            difficulty='medium'
        )
    
    @staticmethod
    def create_alternating_pattern() -> Task:
        """Rozpoznawanie wzoru naprzemiennego"""
        X = np.array([
            [0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1],
            [1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]
        ])
        y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])
        
        return Task(
            name='Alternating Pattern',
            X=X,
            y=y,
            input_dim=3,
            output_dim=1,
            description='Recognize alternating patterns',
            difficulty='medium'
        )
    
    @staticmethod
    def create_repeating_pattern() -> Task:
        """Rozpoznawanie powtarzających się wzorów"""
        X = np.array([
            [0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1],
            [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]
        ])
        y = np.array([[1], [1], [0], [0], [0], [0], [1], [1]])
        
        return Task(
            name='Repeating Pattern',
            X=X,
            y=y,
            input_dim=3,
            output_dim=1,
            description='Recognize repeating patterns',
            difficulty='medium'
        )
    
    @staticmethod
    def create_symmetric_pattern() -> Task:
        """Rozpoznawanie wzorów symetrycznych"""
        X = np.array([
            [0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 1],
            [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1]
        ])
        y = np.array([[1], [1], [1], [1], [0], [0], [0], [0]])
        
        return Task(
            name='Symmetric Pattern',
            X=X,
            y=y,
            input_dim=3,
            output_dim=1,
            description='Recognize symmetric patterns',
            difficulty='medium'
        )
    
    @staticmethod
    def create_palindrome() -> Task:
        """Sprawdzanie palindromów"""
        X = np.array([
            [0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 1],
            [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1]
        ])
        y = np.array([[1], [1], [1], [1], [0], [0], [0], [0]])
        
        return Task(
            name='Palindrome',
            X=X,
            y=y,
            input_dim=3,
            output_dim=1,
            description='Check if pattern is palindrome',
            difficulty='hard'
        )
    
    @staticmethod
    def create_monotonic_sequence() -> Task:
        """Rozpoznawanie ciągów monotonicznych"""
        X = np.array([
            [0, 1, 2], [2, 1, 0], [0, 0, 1], [1, 1, 0],
            [1, 2, 3], [3, 2, 1], [0, 2, 1], [1, 3, 2]
        ])
        y = np.array([[1], [1], [0], [0], [1], [1], [0], [0]])
        
        return Task(
            name='Monotonic Sequence',
            X=X,
            y=y,
            input_dim=3,
            output_dim=1,
            description='Recognize monotonic sequences',
            difficulty='medium'
        )
    
    @staticmethod
    def create_fibonacci_mod2() -> Task:
        """Ciąg Fibonacciego modulo 2"""
        X = np.array([[0], [1], [2], [3], [4], [5], [6], [7]])
        y = np.array([[0], [1], [1], [0], [1], [1], [0], [1]])
        
        return Task(
            name='Fibonacci Mod 2',
            X=X,
            y=y,
            input_dim=1,
            output_dim=1,
            description='Fibonacci sequence modulo 2',
            difficulty='hard'
        )
    
    @staticmethod
    def create_prime_detection() -> Task:
        """Wykrywanie liczb pierwszych"""
        X = np.array([[2], [3], [4], [5], [6], [7], [8], [9], [10], [11]])
        y = np.array([[1], [1], [0], [1], [0], [1], [0], [0], [0], [1]])
        
        return Task(
            name='Prime Detection',
            X=X,
            y=y,
            input_dim=1,
            output_dim=1,
            description='Detect prime numbers',
            difficulty='hard'
        )
    
    @staticmethod
    def create_count_ones() -> Task:
        """Zliczanie jedynek w reprezentacji binarnej"""
        X = np.array([[0], [1], [2], [3], [4], [5], [6], [7]])
        y = np.array([[0], [1], [1], [2], [1], [2], [2], [3]])
        
        return Task(
            name='Count Ones',
            X=X,
            y=y,
            input_dim=1,
            output_dim=1,
            description='Count number of ones in binary representation',
            difficulty='medium'
        )
    
    @staticmethod
    def create_hamming_weight() -> Task:
        """Waga Hamminga (liczba bitów różniących się)"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        return Task(
            name='Hamming Weight',
            X=X,
            y=y,
            input_dim=2,
            output_dim=1,
            description='Hamming weight between two binary numbers',
            difficulty='medium'
        )
    
    @staticmethod
    def create_gray_code() -> Task:
        """Konwersja na kod Graya"""
        X = np.array([[0], [1], [2], [3], [4], [5], [6], [7]])
        y = np.array([[0], [1], [3], [2], [6], [7], [5], [4]])
        
        return Task(
            name='Gray Code',
            X=X,
            y=y,
            input_dim=1,
            output_dim=1,
            description='Convert to Gray code',
            difficulty='hard'
        )
    
    @staticmethod
    def create_boolean_expression() -> Task:
        """Złożone wyrażenie logiczne: (A AND B) OR (C AND NOT D)"""
        X = np.array([
            [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
            [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
            [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
            [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1],
        ])
        y = np.array([[0], [0], [1], [0], [0], [0], [1], [1], 
                     [0], [0], [1], [0], [1], [1], [1], [0]])
        
        return Task(
            name='Boolean Expression',
            X=X,
            y=y,
            input_dim=4,
            output_dim=1,
            description='Complex boolean expression: (A AND B) OR (C AND NOT D)',
            difficulty='hard'
        )
    
    @staticmethod
    def create_circuit_simulation() -> Task:
        """Symulacja prostego układu logicznego"""
        X = np.array([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ])
        y = np.array([[0], [1], [1], [1], [1], [0], [0], [1]])
        
        return Task(
            name='Circuit Simulation',
            X=X,
            y=y,
            input_dim=3,
            output_dim=1,
            description='Simple logic circuit simulation',
            difficulty='hard'
        )
    
    @staticmethod
    def create_state_machine() -> Task:
        """Prosta maszyna stanów"""
        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],
            [0, 0], [0, 1], [1, 0], [1, 1],
        ])
        y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])
        
        return Task(
            name='State Machine',
            X=X,
            y=y,
            input_dim=2,
            output_dim=1,
            description='Simple state machine simulation',
            difficulty='hard'
        )
    
    @staticmethod
    def create_binary_decoder() -> Task:
        """Dekoder binarny 2-na-4"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        
        return Task(
            name='Binary Decoder',
            X=X,
            y=y,
            input_dim=2,
            output_dim=4,
            description='2-to-4 binary decoder',
            difficulty='medium'
        )
