#!/usr/bin/env python3
"""
Persistent Agent System - Prawdziwe uczenie ciągłe
"""

import numpy as np
import pickle
import os
from datetime import datetime
from typing import Dict, Any, Optional
from universal_agent import UniversalAgent
from tasks import TaskFactory
from memory import MyceliumMemory


class PersistentAgent:
    """
    Agent który naprawdę uczy się i pamięta między sesjami
    Singleton pattern - jeden agent na zadanie
    """
    
    _instances = {}  # Cache dla singletonów
    
    def __new__(cls, task_name: str, hidden_dim: int = 4, learning_rate: float = 0.5):
        """Singleton pattern - jeden agent na zadanie"""
        key = f"{task_name}_{hidden_dim}_{learning_rate}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]
    
    def __init__(self, task_name: str, hidden_dim: int = 4, learning_rate: float = 0.5):
        """Inicjalizacja tylko jeśli nowy obiekt"""
        if hasattr(self, '_initialized'):
            return
            
        self.task_name = task_name
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.save_path = f"data/agents/{task_name}_{hidden_dim}_{learning_rate}.pkl"
        
        # Inicjalizuj komponenty
        self.task = TaskFactory.get_task(task_name)
        self.mycelium = MyceliumMemory("../data/mycelium_memory.json")
        
        # Metryki uczenia
        self.training_sessions = []
        self.total_epochs = 0
        self.start_time = datetime.now()
        self.last_accuracy = 0.0
        self.learning_speed = 0.0
        
        # Wczytaj lub stwórz agenta
        self.agent = self._load_or_create_agent()
        
        self._initialized = True
    
    def _load_or_create_agent(self) -> UniversalAgent:
        """Wczytaj istniejącego agenta lub stwórz nowego"""
        try:
            if os.path.exists(self.save_path):
                print(f"📂 Wczytuję agenta {self.task_name} z dysku...")
                with open(self.save_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Odtwórz agenta
                agent = UniversalAgent(
                    task=self.task,
                    hidden_dim=self.hidden_dim,
                    learning_rate=self.learning_rate,
                    initial_weights=data['weights']
                )
                
                # Odtwórz metryki
                agent.loss_history = data.get('loss_history', [])
                agent.accuracy_history = data.get('accuracy_history', [])
                agent.weight_history = data.get('weight_history', [])
                agent.activation_history = data.get('activation_history', [])
                
                # Odtwórz metryki uczenia
                self.training_sessions = data.get('training_sessions', [])
                self.total_epochs = data.get('total_epochs', 0)
                self.start_time = data.get('start_time', datetime.now())
                self.last_accuracy = data.get('last_accuracy', 0.0)
                self.learning_speed = data.get('learning_speed', 0.0)
                
                print(f"✅ Agent wczytany! Historia: {len(self.training_sessions)} sesji, {self.total_epochs} epok")
                return agent
            else:
                print(f"🆕 Tworzę nowego agenta {self.task_name}...")
                return self._create_new_agent()
                
        except Exception as e:
            print(f"❌ Błąd wczytywania agenta: {e}")
            print(f"🆕 Tworzę nowego agenta...")
            return self._create_new_agent()
    
    def _create_new_agent(self) -> UniversalAgent:
        """Stwórz nowego agenta z wagami z Mycelium"""
        # Pobierz wagi z grzybni
        shapes = {
            'W1': (self.task.input_dim, self.hidden_dim),
            'b1': (1, self.hidden_dim),
            'W2': (self.hidden_dim, self.task.output_dim),
            'b2': (1, self.task.output_dim)
        }
        
        initial_weights = self.mycelium.get_fusion_weights(shapes, alpha=0.7)
        
        agent = UniversalAgent(
            task=self.task,
            hidden_dim=self.hidden_dim,
            learning_rate=self.learning_rate,
            initial_weights=initial_weights
        )
        
        return agent
    
    def continue_training(self, epochs: int = 1000) -> Dict[str, Any]:
        """
        Kontynuuj trening agenta (prawdziwe uczenie!)
        """
        print(f"🎓 Kontynuuję trening agenta {self.task_name}...")
        print(f"📊 Historia: {len(self.training_sessions)} sesji, {self.total_epochs} epok")
        
        session_start = datetime.now()
        initial_accuracy = self.last_accuracy
        
        # Trening
        results = self.agent.train(epochs=epochs, verbose=False)
        
        # Aktualizuj metryki
        session_end = datetime.now()
        session_duration = (session_end - session_start).total_seconds()
        
        # Oblicz prędkość uczenia
        accuracy_improvement = results['final_accuracy'] - initial_accuracy
        self.learning_speed = accuracy_improvement / session_duration if session_duration > 0 else 0
        
        # Zapisz sesję
        session_data = {
            'session_id': len(self.training_sessions) + 1,
            'start_time': session_start.isoformat(),
            'end_time': session_end.isoformat(),
            'epochs': epochs,
            'initial_accuracy': initial_accuracy,
            'final_accuracy': results['final_accuracy'],
            'improvement': accuracy_improvement,
            'learning_speed': self.learning_speed,
            'final_loss': results['final_loss'],
            'training_time': results['training_time']
        }
        
        self.training_sessions.append(session_data)
        self.total_epochs += epochs
        self.last_accuracy = results['final_accuracy']
        
        # Próba aktualizacji grzybni
        weights = self.agent.get_weights()
        updated = self.mycelium.update_memory(
            weights=weights,
            loss=results['final_loss'],
            metadata={
                'task_name': self.task_name,
                'accuracy': results['final_accuracy'],
                'total_epochs': self.total_epochs,
                'learning_speed': self.learning_speed,
                'session_count': len(self.training_sessions)
            }
        )
        
        # Zapisz agenta
        self._save_agent()
        
        # Dodaj metryki do wyników
        results.update({
            'session_id': session_data['session_id'],
            'total_epochs': self.total_epochs,
            'session_count': len(self.training_sessions),
            'learning_speed': self.learning_speed,
            'improvement': accuracy_improvement,
            'updated_mycelium': updated,
            'agent': self.agent
        })
        
        print(f"✅ Trening zakończony! Poprawa: {accuracy_improvement:+.3f}, Prędkość: {self.learning_speed:.4f}/s")
        
        return results
    
    def _save_agent(self):
        """Zapisz stan agenta na dysku"""
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            data = {
                'weights': self.agent.get_weights(),
                'loss_history': self.agent.loss_history,
                'accuracy_history': self.agent.accuracy_history,
                'weight_history': self.agent.weight_history,
                'activation_history': self.agent.activation_history,
                'training_sessions': self.training_sessions,
                'total_epochs': self.total_epochs,
                'start_time': self.start_time,
                'last_accuracy': self.last_accuracy,
                'learning_speed': self.learning_speed,
                'save_time': datetime.now().isoformat()
            }
            
            with open(self.save_path, 'wb') as f:
                pickle.dump(data, f)
                
            print(f"💾 Agent {self.task_name} zapisany!")
            
        except Exception as e:
            print(f"❌ Błąd zapisu agenta: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Zwrósz szczegółowe statystyki uczenia"""
        if not self.training_sessions:
            return {
                'total_sessions': 0,
                'total_epochs': 0,
                'total_time': 0,
                'average_speed': 0,
                'best_accuracy': 0,
                'current_accuracy': 0,
                'improvement_trend': 'none'
            }
        
        # Oblicz statystyki
        total_time = sum(s['training_time'] for s in self.training_sessions)
        average_speed = np.mean([s['learning_speed'] for s in self.training_sessions])
        best_accuracy = max(s['final_accuracy'] for s in self.training_sessions)
        
        # Trend poprawy
        if len(self.training_sessions) >= 3:
            recent_improvements = [s['improvement'] for s in self.training_sessions[-3:]]
            avg_improvement = np.mean(recent_improvements)
            if avg_improvement > 0.01:
                trend = 'improving'
            elif avg_improvement < -0.01:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'total_sessions': len(self.training_sessions),
            'total_epochs': self.total_epochs,
            'total_time': total_time,
            'average_speed': average_speed,
            'best_accuracy': best_accuracy,
            'current_accuracy': self.last_accuracy,
            'improvement_trend': trend,
            'first_accuracy': self.training_sessions[0]['final_accuracy'] if self.training_sessions else 0,
            'total_improvement': self.last_accuracy - (self.training_sessions[0]['final_accuracy'] if self.training_sessions else 0)
        }
    
    def reset(self):
        """Resetuj agenta do stanu początkowego"""
        print(f"🔄 Resetuję agenta {self.task_name}...")
        
        # Usuń plik
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        
        # Resetuj metryki
        self.training_sessions = []
        self.total_epochs = 0
        self.start_time = datetime.now()
        self.last_accuracy = 0.0
        self.learning_speed = 0.0
        
        # Stwórz nowego agenta
        self.agent = self._create_new_agent()
        
        print(f"✅ Agent {self.task_name} zresetowany!")


def get_persistent_agent(task_name: str, hidden_dim: int = 4, learning_rate: float = 0.5) -> PersistentAgent:
    """Pobierz lub stwórz persistent agenta"""
    return PersistentAgent(task_name, hidden_dim, learning_rate)


def list_all_agents() -> Dict[str, Dict[str, Any]]:
    """Lista wszystkich zapisanych agentów"""
    agents_dir = "data/agents"
    agents = {}
    
    if os.path.exists(agents_dir):
        for filename in os.listdir(agents_dir):
            if filename.endswith('.pkl'):
                try:
                    filepath = os.path.join(agents_dir, filename)
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Wyodrębnij metadane
                    task_name = filename.split('_')[0]
                    agents[task_name] = {
                        'file': filename,
                        'total_epochs': data.get('total_epochs', 0),
                        'sessions': len(data.get('training_sessions', [])),
                        'last_accuracy': data.get('last_accuracy', 0),
                        'learning_speed': data.get('learning_speed', 0),
                        'save_time': data.get('save_time', 'Unknown')
                    }
                except Exception as e:
                    print(f"Błąd wczytywania agenta {filename}: {e}")
    
    return agents
