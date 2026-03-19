import numpy as np
import time
from typing import Tuple, List, Dict, Any, Optional
import json
from datetime import datetime

from tasks import Task, TaskFactory


class UniversalAgent:
    """
    Uniwersalny agent uczący się różnych zadań.
    Automatycznie dostosowuje architekturę do wymiaru zadania.
    """
    
    def __init__(
        self, 
        task: Task,
        hidden_dim: int = 4, 
        learning_rate: float = 0.5, 
        initial_weights: Optional[Dict[str, np.ndarray]] = None
    ):
        self.task = task
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.input_dim = task.input_dim
        self.output_dim = task.output_dim
        
        # Dane treningowe z zadania
        self.X = task.X
        self.y = task.y
        
        # Inicjalizacja wag
        if initial_weights is not None:
            self.W1 = initial_weights['W1'].copy()
            self.b1 = initial_weights['b1'].copy()
            self.W2 = initial_weights['W2'].copy()
            self.b2 = initial_weights['b2'].copy()
        else:
            # Xavier initialization
            limit1 = np.sqrt(6 / (self.input_dim + hidden_dim))
            limit2 = np.sqrt(6 / (hidden_dim + self.output_dim))
            
            self.W1 = np.random.uniform(-limit1, limit1, (self.input_dim, hidden_dim))
            self.b1 = np.zeros((1, hidden_dim))
            self.W2 = np.random.uniform(-limit2, limit2, (hidden_dim, self.output_dim))
            self.b2 = np.zeros((1, self.output_dim))
        
        # Historia treningu
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []
        self.epoch_times: List[float] = []
        
    def get_weight_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Zwraca kształty wag dla tego zadania"""
        return {
            'W1': (self.input_dim, self.hidden_dim),
            'b1': (1, self.hidden_dim),
            'W2': (self.hidden_dim, self.output_dim),
            'b2': (1, self.output_dim)
        }
    
    def relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)
    
    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1.0, 0.0)
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a: np.ndarray) -> np.ndarray:
        return a * (1 - a)
    
    def forward_pass(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Propagacja w przód"""
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)
        return z1, a1, z2, a2
    
    def calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary cross-entropy loss"""
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(loss)
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Oblicza dokładność (próg 0.5)"""
        predictions = (y_pred > 0.5).astype(int)
        return float(np.mean(predictions == y_true))
    
    def backward_pass(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        z1: np.ndarray, 
        a1: np.ndarray, 
        a2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Propagacja wsteczna"""
        d2 = a2 - y
        dW2 = a1.T @ d2
        db2 = np.sum(d2, axis=0, keepdims=True)
        
        d1 = d2 @ self.W2.T * self.relu_derivative(z1)
        dW1 = X.T @ d1
        db1 = np.sum(d1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1: np.ndarray, db1: np.ndarray, dW2: np.ndarray, db2: np.ndarray):
        """Aktualizacja wag"""
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, epochs: int = 5000, verbose: bool = True, save_interval: int = 500) -> Dict[str, Any]:
        """
        Trening agenta.
        
        Returns:
            Dict z wynikami treningu
        """
        if verbose:
            print(f"=== TRENING: {self.task.name} ===")
            print(f"Epoki: {epochs}, Learning rate: {self.learning_rate}")
            print(f"Architektura: {self.input_dim}→{self.hidden_dim}→{self.output_dim}")
            print(f"Trudność: {self.task.difficulty}")
        
        start_time = time.time()
        best_accuracy = 0.0
        best_loss = float('inf')
        patience = 2000
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Forward pass
            z1, a1, z2, a2 = self.forward_pass(self.X)
            
            # Obliczanie metryk
            loss = self.calculate_loss(self.y, a2)
            accuracy = self.calculate_accuracy(self.y, a2)
            
            # Zapis historii
            if epoch % 100 == 0:
                self.loss_history.append(loss)
                self.accuracy_history.append(accuracy)
                self.epoch_times.append(time.time() - epoch_start)
            
            # Logowanie
            if verbose and epoch % save_interval == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f} | Acc: {accuracy:.3f}")
            
            # Early stopping check
            if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
                best_accuracy = accuracy
                best_loss = loss
                patience_counter = 0
                # Zapis najlepszych wag
                best_W1 = self.W1.copy()
                best_W2 = self.W2.copy()
                best_b1 = self.b1.copy()
                best_b2 = self.b2.copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience and accuracy >= 0.95:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                # Przywróć najlepsze wagi
                self.W1 = best_W1
                self.W2 = best_W2
                self.b1 = best_b1
                self.b2 = best_b2
                break
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward_pass(self.X, self.y, z1, a1, a2)
            
            # Aktualizacja wag
            self.update_weights(dW1, db1, dW2, db2)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Finalna ewaluacja
        z1, a1, z2, a2 = self.forward_pass(self.X)
        final_loss = self.calculate_loss(self.y, a2)
        final_accuracy = self.calculate_accuracy(self.y, a2)
        
        if verbose:
            print(f"\n=== WYNIKI ===")
            print(f"Czas treningu: {training_time:.2f}s")
            print(f"Finalna dokładność: {final_accuracy:.3f}")
            print(f"Finalna strata: {final_loss:.6f}")
            
            if final_accuracy >= 0.95:
                print(f"✅ Agent nauczył się zadania {self.task.name}!")
            else:
                print(f"⚠️  Agent potrzebuje więcej treningu")
        
        return {
            'final_loss': final_loss,
            'final_accuracy': final_accuracy,
            'training_time': training_time,
            'epochs_trained': len(self.loss_history) * 100,
            'task_name': self.task.name,
            'task_difficulty': self.task.difficulty
        }
    
    def evaluate(self, verbose: bool = True) -> Dict[str, Any]:
        """Ewaluacja agenta na danych treningowych"""
        z1, a1, z2, a2 = self.forward_pass(self.X)
        loss = self.calculate_loss(self.y, a2)
        accuracy = self.calculate_accuracy(self.y, a2)
        predictions = (a2 > 0.5).astype(int)
        
        if verbose:
            print(f"\n=== EWALUACJA: {self.task.name} ===")
            print("Wejścia:")
            print(self.X)
            print("\nPredykcje (surowe):")
            print(np.round(a2, 4))
            print("\nPredykcje (binarne):")
            print(predictions)
            print("\nOczekiwane:")
            print(self.y)
            print(f"\nDokładność: {accuracy:.3f}")
            print(f"Strata: {loss:.6f}")
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': predictions,
            'raw_predictions': a2
        }
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Zwraca wagi agenta"""
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }


if __name__ == "__main__":
    print("🧪 Test UniversalAgent na różnych zadaniach\n")
    
    # Test na prostych zadaniach
    easy_tasks = ['and', 'or', 'nand']
    
    for task_name in easy_tasks:
        print("=" * 60)
        task = TaskFactory.get_task(task_name)
        agent = UniversalAgent(task, hidden_dim=4, learning_rate=0.5)
        
        results = agent.train(epochs=2000, verbose=True, save_interval=500)
        agent.evaluate(verbose=True)
        print()
    
    # Test na trudniejszym zadaniu
    print("=" * 60)
    print("\n🔥 TRUDNIEJSZE ZADANIE: 3-bit Parity\n")
    parity_task = TaskFactory.get_task('3bit_parity')
    parity_agent = UniversalAgent(parity_task, hidden_dim=8, learning_rate=0.5)
    
    results = parity_agent.train(epochs=10000, verbose=True, save_interval=2000)
    parity_agent.evaluate(verbose=True)
