#!/usr/bin/env python3
"""
Auto Evolution System - Autonomiczny trening agentów
"""

import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
from persistent_agent import get_persistent_agent
from tasks import TaskFactory
import json


class AutoEvolutionSystem:
    """
    System który automatycznie trenuje agentów na wszystkich zadaniach
    z optymalizacją parametrów i śledzeniem postępów
    """
    
    def __init__(self):
        self.tasks = TaskFactory.list_tasks()
        self.evolution_log = []
        self.current_task_index = 0
        self.mastered_tasks = set()
        self.failed_attempts = {}
        self.optimization_history = {}
        
        # Strategie uczenia dla różnych zadań
        self.task_strategies = {
            'and': {'base_epochs': 1000, 'hidden_dims': [4, 6, 8], 'learning_rates': [0.3, 0.5, 0.7]},
            'or': {'base_epochs': 1000, 'hidden_dims': [4, 6, 8], 'learning_rates': [0.3, 0.5, 0.7]},
            'not': {'base_epochs': 800, 'hidden_dims': [3, 4, 5], 'learning_rates': [0.3, 0.5, 0.7]},
            'xor': {'base_epochs': 2000, 'hidden_dims': [4, 6, 8, 10], 'learning_rates': [0.2, 0.4, 0.6]},
            'nand': {'base_epochs': 2000, 'hidden_dims': [4, 6, 8, 10], 'learning_rates': [0.2, 0.4, 0.6]},
            'nor': {'base_epochs': 2000, 'hidden_dims': [4, 6, 8, 10], 'learning_rates': [0.2, 0.4, 0.6]}
        }
        
        # Cele ewolucji
        self.mastery_threshold = 0.95
        self.max_attempts_per_task = 5
        self.global_start_time = datetime.now()
    
    def run_full_evolution(self, progress_callback=None):
        """
        Uruchom pełną ewolucję - automatyczne uczenie wszystkich zadań
        """
        print("🚀 Rozpoczynam Auto-Evolution System!")
        print(f"📋 Zadania do nauki: {list(self.tasks.keys())}")
        print(f"🎯 Cel: {self.mastery_threshold*100}% accuracy na wszystkich zadaniach")
        
        evolution_results = {
            'start_time': self.global_start_time.isoformat(),
            'tasks_mastered': [],
            'tasks_failed': [],
            'total_time': 0,
            'total_sessions': 0,
            'evolution_log': []
        }
        
        # Przejdz przez wszystkie zadania
        for task_name in self.tasks.keys():
            if progress_callback:
                progress_callback(f"🎯 Pracuję nad zadaniem: {task_name.upper()}")
            
            task_result = self._master_task(task_name, progress_callback)
            evolution_results['evolution_log'].append(task_result)
            
            if task_result['success']:
                evolution_results['tasks_mastered'].append(task_name)
                self.mastered_tasks.add(task_name)
                if progress_callback:
                    progress_callback(f"✅ Zadanie {task_name.upper()} opanowane!")
            else:
                evolution_results['tasks_failed'].append(task_name)
                if progress_callback:
                    progress_callback(f"❌ Zadanie {task_name.upper()} nieudane")
        
        # Podsumowanie
        evolution_results['end_time'] = datetime.now().isoformat()
        evolution_results['total_time'] = (datetime.now() - self.global_start_time).total_seconds()
        evolution_results['total_sessions'] = sum(len(log['attempts']) for log in evolution_results['evolution_log'])
        
        # Zapisz wyniki
        self._save_evolution_results(evolution_results)
        
        return evolution_results
    
    def _master_task(self, task_name: str, progress_callback=None) -> Dict[str, Any]:
        """
        Opanuj pojedyncze zadanie przez optymalizację parametrów
        """
        print(f"🎯 Rozpoczynam opanowywanie zadania: {task_name}")
        
        task_result = {
            'task_name': task_name,
            'success': False,
            'best_accuracy': 0.0,
            'attempts': [],
            'total_time': 0,
            'optimal_params': None
        }
        
        strategy = self.task_strategies.get(task_name, {
            'base_epochs': 1500,
            'hidden_dims': [4, 6, 8],
            'learning_rates': [0.3, 0.5, 0.7]
        })
        
        attempts = 0
        task_start_time = datetime.now()
        
        # Próbuj różnych konfiguracji
        for hidden_dim in strategy['hidden_dims']:
            for learning_rate in strategy['learning_rates']:
                if attempts >= self.max_attempts_per_task:
                    break
                
                attempts += 1
                
                if progress_callback:
                    progress_callback(f"🔄 {task_name.upper()} - Próba {attempts}/{self.max_attempts_per_task}: H={hidden_dim}, LR={learning_rate}")
                
                # Oblicz epoki na podstawie trudności
                epochs = self._calculate_epochs(task_name, hidden_dim, learning_rate, attempts)
                
                attempt_result = self._train_single_attempt(
                    task_name, hidden_dim, learning_rate, epochs, progress_callback
                )
                
                task_result['attempts'].append(attempt_result)
                
                # Sprawdź czy osiągnięto mistrzostwo
                if attempt_result['final_accuracy'] >= self.mastery_threshold:
                    task_result['success'] = True
                    task_result['best_accuracy'] = attempt_result['final_accuracy']
                    task_result['optimal_params'] = {
                        'hidden_dim': hidden_dim,
                        'learning_rate': learning_rate,
                        'epochs': epochs
                    }
                    print(f"🎉 {task_name.upper()} opanowane! Accuracy: {attempt_result['final_accuracy']:.3f}")
                    break
                
                # Aktualizuj najlepszy wynik
                if attempt_result['final_accuracy'] > task_result['best_accuracy']:
                    task_result['best_accuracy'] = attempt_result['final_accuracy']
                    task_result['optimal_params'] = {
                        'hidden_dim': hidden_dim,
                        'learning_rate': learning_rate,
                        'epochs': epochs
                    }
        
        task_result['total_time'] = (datetime.now() - task_start_time).total_seconds()
        
        print(f"📊 {task_name.upper()} - Najlepszy wynik: {task_result['best_accuracy']:.3f}")
        print(f"⏱️ Czas: {task_result['total_time']:.1f}s, Próby: {len(task_result['attempts'])}")
        
        return task_result
    
    def _calculate_epochs(self, task_name: str, hidden_dim: int, learning_rate: float, attempt: int) -> int:
        """
        Inteligentne obliczanie liczby epok na podstawie parametrów
        """
        base_strategy = self.task_strategies.get(task_name, {'base_epochs': 1500})
        base_epochs = base_strategy['base_epochs']
        
        # Modyfikatory
        dim_modifier = 1.0 + (hidden_dim - 4) * 0.1  # Więcej neuronów = więcej epok
        lr_modifier = 1.0 / learning_rate  # Niższy LR = więcej epok
        attempt_modifier = 1.0 + (attempt - 1) * 0.2  # Późniejsze próby = więcej epok
        
        calculated_epochs = int(base_epochs * dim_modifier * lr_modifier * attempt_modifier)
        
        # Ograniczenia
        min_epochs = 500
        max_epochs = 10000
        
        return max(min_epochs, min(max_epochs, calculated_epochs))
    
    def _train_single_attempt(self, task_name: str, hidden_dim: int, learning_rate: float, epochs: int, progress_callback=None) -> Dict[str, Any]:
        """
        Pojedyncza próba treningu
        """
        attempt_start = datetime.now()
        
        try:
            # Pobierz agenta
            agent = get_persistent_agent(task_name, hidden_dim, learning_rate)
            
            # Trening
            if progress_callback:
                progress_callback(f"🏃 Trenuję {task_name.upper()}: {epochs} epok...")
            
            results = agent.continue_training(epochs=epochs)
            
            # Dodatkowe statystyki
            learning_stats = agent.get_learning_stats()
            
            attempt_result = {
                'attempt_id': f"{task_name}_{hidden_dim}_{learning_rate}_{datetime.now().strftime('%H%M%S')}",
                'hidden_dim': hidden_dim,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'final_accuracy': results['final_accuracy'],
                'final_loss': results['final_loss'],
                'training_time': results['training_time'],
                'total_sessions': learning_stats['total_sessions'],
                'learning_speed': learning_stats['average_speed'],
                'improvement_trend': learning_stats['improvement_trend'],
                'success': results['final_accuracy'] >= self.mastery_threshold,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"📈 {task_name.upper()} H{hidden_dim} LR{learning_rate}: {results['final_accuracy']:.3f} ({results['training_time']:.1f}s)")
            
            return attempt_result
            
        except Exception as e:
            print(f"❌ Błąd w próbie {task_name}: {e}")
            return {
                'attempt_id': f"{task_name}_{hidden_dim}_{learning_rate}_error",
                'hidden_dim': hidden_dim,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'final_accuracy': 0.0,
                'final_loss': float('inf'),
                'training_time': 0.0,
                'total_sessions': 0,
                'learning_speed': 0.0,
                'improvement_trend': 'error',
                'success': False,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Aktualny status ewolucji
        """
        return {
            'tasks_count': len(self.tasks),
            'mastered_count': len(self.mastered_tasks),
            'mastered_tasks': list(self.mastered_tasks),
            'remaining_tasks': [task for task in self.tasks.keys() if task not in self.mastered_tasks],
            'progress_percentage': (len(self.mastered_tasks) / len(self.tasks)) * 100,
            'current_time': datetime.now().isoformat()
        }
    
    def _save_evolution_results(self, results: Dict[str, Any]):
        """Zapisz wyniki ewolucji"""
        try:
            import os
            os.makedirs('data/evolution', exist_ok=True)
            
            filename = f"data/evolution/evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Wyniki ewolucji zapisane: {filename}")
            
        except Exception as e:
            print(f"❌ Błąd zapisu wyników: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Podsumowanie optymalizacji parametrów
        """
        summary = {}
        
        for task_name in self.tasks.keys():
            task_attempts = [log for log in self.evolution_log if log['task_name'] == task_name]
            if task_attempts:
                best_attempt = max(task_attempts, key=lambda x: x['best_accuracy'])
                summary[task_name] = {
                    'best_accuracy': best_attempt['best_accuracy'],
                    'optimal_params': best_attempt['optimal_params'],
                    'attempts_needed': len(best_attempt['attempts']),
                    'total_time': best_attempt['total_time']
                }
        
        return summary


def run_auto_evolution(progress_callback=None):
    """
    Funkcja convenience do uruchomienia auto-evolution
    """
    system = AutoEvolutionSystem()
    return system.run_full_evolution(progress_callback)


def get_evolution_status():
    """
    Pobierz aktualny status ewolucji
    """
    system = AutoEvolutionSystem()
    return system.get_current_status()
