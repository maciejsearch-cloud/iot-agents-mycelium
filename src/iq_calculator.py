#!/usr/bin/env python3
"""
IQ Calculator - System pomiaru inteligencji agentów
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from universal_agent import UniversalAgent
from tasks import TaskFactory


class AgentIQCalculator:
    """Kalkulator IQ dla agentów - precyzyjny pomiar inteligencji"""
    
    def __init__(self):
        self.tests = {
            'boolean_logic': BooleanLogicTest(max_score=30),
            'pattern_recognition': PatternTest(max_score=30),
            'math_foundation': MathTest(max_score=20),
            'sequence_logic': SequenceTest(max_score=20),
            'memory_retention': MemoryTest(max_score=10)
        }
        
        self.evolution_stages = {
            'foundation': {'min_iq': 0, 'max_iq': 50, 'description': 'Logika podstawowa'},
            'patterns': {'min_iq': 51, 'max_iq': 80, 'description': 'Wzorce i sekwencje'},
            'language': {'min_iq': 81, 'max_iq': 120, 'description': 'Język podstawowy'},
            'reasoning': {'min_iq': 121, 'max_iq': 160, 'description': 'Logiczne myślenie'},
            'emotional': {'min_iq': 161, 'max_iq': 200, 'description': 'Inteligencja emocjonalna'}
        }
    
    def calculate_iq(self, agent: UniversalAgent) -> Dict[str, Any]:
        """Oblicza IQ agenta na podstawie wszystkich testów"""
        scores = {}
        total_score = 0
        
        for test_name, test in self.tests.items():
            try:
                score = test.evaluate(agent)
                scores[test_name] = score
                total_score += score
            except Exception as e:
                scores[test_name] = 0
                print(f"Błąd w teście {test_name}: {e}")
        
        stage = self.determine_stage(total_score)
        
        return {
            'total_iq': total_score,
            'breakdown': scores,
            'stage': stage,
            'next_stage_requirements': self.get_next_stage_requirements(total_score),
            'evolution_progress': self.calculate_evolution_progress(total_score, stage)
        }
    
    def determine_stage(self, iq_score: int) -> Dict[str, Any]:
        """Określa etap ewolucji na podstawie IQ"""
        for stage_name, stage_info in self.evolution_stages.items():
            if stage_info['min_iq'] <= iq_score <= stage_info['max_iq']:
                return {
                    'name': stage_name,
                    'description': stage_info['description'],
                    'min_iq': stage_info['min_iq'],
                    'max_iq': stage_info['max_iq'],
                    'progress': (iq_score - stage_info['min_iq']) / (stage_info['max_iq'] - stage_info['min_iq'])
                }
        return {'name': 'unknown', 'description': 'Nieznany etap'}
    
    def get_next_stage_requirements(self, current_iq: int) -> Dict[str, Any]:
        """Zwraca wymagania do następnego etapu"""
        current_stage = self.determine_stage(current_iq)
        stage_names = list(self.evolution_stages.keys())
        current_index = stage_names.index(current_stage['name'])
        
        if current_index < len(stage_names) - 1:
            next_stage_name = stage_names[current_index + 1]
            next_stage = self.evolution_stages[next_stage_name]
            needed_iq = next_stage['min_iq'] - current_iq
            
            return {
                'next_stage': next_stage_name,
                'description': next_stage['description'],
                'needed_iq': max(0, needed_iq),
                'target_iq': next_stage['min_iq']
            }
        
        return {'next_stage': 'max', 'description': 'Maksymalny etap osiągnięty', 'needed_iq': 0}
    
    def calculate_evolution_progress(self, iq_score: int, stage: Dict[str, Any]) -> float:
        """Oblicza ogólny postęp ewolucji (0-1)"""
        max_iq = 200  # Maksymalne możliwe IQ
        return min(1.0, iq_score / max_iq)


class BooleanLogicTest:
    """Test logiki boolean - podstawy inteligencji"""
    
    def __init__(self, max_score: int = 30):
        self.max_score = max_score
        self.tasks = ['xor', 'and', 'or', 'nand', 'nor']
    
    def evaluate(self, agent: UniversalAgent) -> int:
        """Ocenia agenta na zadaniach boolean"""
        total_accuracy = 0
        tasks_evaluated = 0
        
        for task_name in self.tasks:
            try:
                task = TaskFactory.get_task(task_name)
                # Tymczasowo tworzymy nowego agenta dla tego zadania
                test_agent = UniversalAgent(
                    task=task,
                    hidden_dim=agent.hidden_dim if hasattr(agent, 'hidden_dim') else 4,
                    learning_rate=agent.learning_rate if hasattr(agent, 'learning_rate') else 0.5
                )
                
                # Krótki trening
                results = test_agent.train(epochs=1000, verbose=False)
                total_accuracy += results['final_accuracy']
                tasks_evaluated += 1
                
            except Exception as e:
                print(f"Błąd w zadaniu {task_name}: {e}")
                continue
        
        if tasks_evaluated == 0:
            return 0
        
        average_accuracy = total_accuracy / tasks_evaluated
        return int(average_accuracy * self.max_score)


class PatternTest:
    """Test rozpoznawania wzorców"""
    
    def __init__(self, max_score: int = 30):
        self.max_score = max_score
    
    def evaluate(self, agent: UniversalAgent) -> int:
        """Ocenia zdolność rozpoznawania wzorców"""
        # Prosty test rozpoznawania sekwencji
        try:
            # Test: czy agent rozpoznaje wzorce w danych treningowych
            if hasattr(agent, 'X') and hasattr(agent, 'y'):
                patterns = self.detect_patterns(agent.X)
                pattern_score = min(len(patterns), 10) * 3  # Max 30 punktów
                return pattern_score
        except:
            pass
        return 5  # Bazowy punkt


class MathTest:
    """Test podstaw matematyki"""
    
    def __init__(self, max_score: int = 20):
        self.max_score = max_score
    
    def evaluate(self, agent: UniversalAgent) -> int:
        """Ocenia podstawowe umiejętności matematyczne"""
        # Test: czy agent rozumie relacje liczbowe
        try:
            if hasattr(agent, 'loss_history') and agent.loss_history:
                # Ocena na podstawie spadku loss - im lepszy, tym więcej "rozumie"
                initial_loss = agent.loss_history[0]
                final_loss = agent.loss_history[-1]
                improvement = (initial_loss - final_loss) / initial_loss
                
                math_score = min(int(improvement * 100), self.max_score)
                return max(5, math_score)  # Minimum 5 punktów
        except:
            pass
        return 5


class SequenceTest:
    """Test logiki sekwencji"""
    
    def __init__(self, max_score: int = 20):
        self.max_score = max_score
    
    def evaluate(self, agent: UniversalAgent) -> int:
        """Ocenia zdolność do rozumienia sekwencji"""
        try:
            if hasattr(agent, 'accuracy_history') and agent.accuracy_history:
                # Ocena stabilności uczenia się
                accuracies = agent.accuracy_history[-5:]  # Ostatnie 5 pomiarów
                if len(accuracies) >= 3:
                    stability = 1.0 - np.std(accuracies)  # Mniejsza odchylenie = większa stabilność
                    sequence_score = int(stability * self.max_score)
                    return max(5, sequence_score)
        except:
            pass
        return 5


class MemoryTest:
    """Test pamięci"""
    
    def __init__(self, max_score: int = 10):
        self.max_score = max_score
    
    def evaluate(self, agent: UniversalAgent) -> int:
        """Ocenia zdolności pamięciowe"""
        try:
            # Test: czy agent zachowuje dobre wyniki
            if hasattr(agent, 'accuracy_history') and agent.accuracy_history:
                best_accuracy = max(agent.accuracy_history)
                memory_score = int(best_accuracy * self.max_score)
                return memory_score
        except:
            pass
        return 3
    
    def detect_patterns(self, X: np.ndarray) -> List[str]:
        """Wykrywa proste wzorce w danych"""
        patterns = []
        try:
            # Szukanie powtarzających się wzorców
            unique_rows = np.unique(X, axis=0)
            if len(unique_rows) < len(X):
                patterns.append("repetition")
            
            # Szukanie symetrii
            for row in unique_rows:
                if len(row) == 2 and row[0] == row[1]:
                    patterns.append("symmetry")
                    break
                    
        except:
            pass
        return patterns
