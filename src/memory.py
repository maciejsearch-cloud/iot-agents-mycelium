import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from filelock import FileLock
import os


class MyceliumMemory:
    """
    Wspólna pamięć agentów (Mycelium Network) - thread-safe i process-safe.
    Przechowuje najlepsze wagi z najniższym loss, umożliwiając transfer wiedzy między agentami.
    """
    
    def __init__(self, memory_path: str = "../data/mycelium_memory.json"):
        self.memory_path = Path(memory_path).resolve()
        self.lock_path = Path(str(self.memory_path) + ".lock")
        
        # Upewnij się że katalog istnieje
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Inicjalizuj pusty plik jeśli nie istnieje
        if not self.memory_path.exists():
            self._initialize_empty_memory()
    
    def _initialize_empty_memory(self) -> None:
        """Tworzy pusty plik pamięci"""
        empty_memory = {
            "best_loss": float('inf'),
            "best_weights": None,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": None,
                "total_updates": 0,
                "best_agent_id": None
            }
        }
        
        with FileLock(self.lock_path, timeout=10):
            with open(self.memory_path, 'w') as f:
                json.dump(empty_memory, f, indent=2)
    
    def _read_memory(self) -> Dict[str, Any]:
        """Odczytuje pamięć z pliku (thread-safe)"""
        with FileLock(self.lock_path, timeout=10):
            if not self.memory_path.exists():
                self._initialize_empty_memory()
            
            with open(self.memory_path, 'r') as f:
                return json.load(f)
    
    def _write_memory(self, data: Dict[str, Any]) -> None:
        """Zapisuje pamięć do pliku (thread-safe)"""
        with FileLock(self.lock_path, timeout=10):
            with open(self.memory_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def get_fusion_weights(
        self, 
        shapes: Dict[str, Tuple[int, ...]], 
        alpha: float = 0.7
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Pobiera wagi z grzybni i miesza je z nowymi losowymi wagami.
        
        Args:
            shapes: Słownik z kształtami wag {'W1': (2, 4), 'b1': (1, 4), ...}
            alpha: Współczynnik mieszania (0-1). 
                   alpha=1.0 -> tylko wagi z grzybni
                   alpha=0.0 -> tylko losowe wagi
        
        Returns:
            Słownik z wagami lub None jeśli grzybnia pusta
        
        Wzór: W_nowy = (W_grzybni * alpha) + (W_losowe * (1 - alpha))
        """
        memory = self._read_memory()
        
        # Jeśli grzybnia pusta, zwróć losowe wagi
        if memory["best_weights"] is None:
            return self._generate_random_weights(shapes)
        
        # Pobierz wagi z grzybni
        mycelium_weights = {
            key: np.array(value) 
            for key, value in memory["best_weights"].items()
        }
        
        # Wygeneruj losowe wagi (Xavier initialization)
        random_weights = self._generate_random_weights(shapes)
        
        # Mieszaj wagi: fusion = mycelium * alpha + random * (1 - alpha)
        fusion_weights = {}
        for key in shapes.keys():
            if key in mycelium_weights:
                fusion_weights[key] = (
                    mycelium_weights[key] * alpha + 
                    random_weights[key] * (1 - alpha)
                )
            else:
                # Jeśli brak wagi w grzybni, użyj losowej
                fusion_weights[key] = random_weights[key]
        
        return fusion_weights
    
    def _generate_random_weights(
        self, 
        shapes: Dict[str, Tuple[int, ...]]
    ) -> Dict[str, np.ndarray]:
        """Generuje losowe wagi z Xavier initialization"""
        weights = {}
        
        for key, shape in shapes.items():
            if key.startswith('W'):  # Wagi (nie biasy)
                # Xavier/Glorot initialization
                if len(shape) == 2:
                    fan_in, fan_out = shape
                    limit = np.sqrt(6 / (fan_in + fan_out))
                    weights[key] = np.random.uniform(-limit, limit, shape)
                else:
                    weights[key] = np.random.randn(*shape) * 0.1
            else:  # Biasy
                weights[key] = np.zeros(shape)
        
        return weights
    
    def update_memory(
        self, 
        weights: Dict[str, np.ndarray], 
        loss: float, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Aktualizuje pamięć grzybni tylko jeśli nowy loss jest lepszy.
        
        Args:
            weights: Słownik z wagami {'W1': array, 'b1': array, ...}
            loss: Wartość straty
            metadata: Dodatkowe metadane (agent_id, accuracy, itp.)
        
        Returns:
            True jeśli pamięć została zaktualizowana, False w przeciwnym razie
        """
        memory = self._read_memory()
        
        # Sprawdź czy nowy loss jest lepszy
        if loss >= memory["best_loss"]:
            return False
        
        # Konwertuj numpy arrays na listy dla JSON
        weights_serializable = {
            key: value.tolist() 
            for key, value in weights.items()
        }
        
        # Aktualizuj pamięć
        memory["best_loss"] = float(loss)
        memory["best_weights"] = weights_serializable
        memory["metadata"]["last_updated"] = datetime.now().isoformat()
        memory["metadata"]["total_updates"] += 1
        
        # Dodaj custom metadata jeśli podano
        if metadata:
            for key, value in metadata.items():
                memory["metadata"][key] = value
        
        # Zapisz do pliku
        self._write_memory(memory)
        
        return True
    
    def get_best_loss(self) -> float:
        """Zwraca najlepszy loss z grzybni"""
        memory = self._read_memory()
        return memory["best_loss"]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Zwraca metadane grzybni"""
        memory = self._read_memory()
        return memory["metadata"]
    
    def is_empty(self) -> bool:
        """Sprawdza czy grzybnia jest pusta"""
        memory = self._read_memory()
        return memory["best_weights"] is None
    
    def reset(self) -> None:
        """Resetuje pamięć grzybni"""
        self._initialize_empty_memory()
    
    def get_stats(self) -> Dict[str, Any]:
        """Zwraca statystyki grzybni"""
        memory = self._read_memory()
        
        return {
            "is_empty": memory["best_weights"] is None,
            "best_loss": memory["best_loss"],
            "total_updates": memory["metadata"]["total_updates"],
            "last_updated": memory["metadata"]["last_updated"],
            "created_at": memory["metadata"]["created_at"]
        }


if __name__ == "__main__":
    # Test MyceliumMemory
    print("🍄 Test Mycelium Memory")
    print("=" * 50)
    
    # Inicjalizacja
    mycelium = MyceliumMemory("../data/test_mycelium.json")
    
    print(f"\n1. Sprawdzanie czy grzybnia pusta: {mycelium.is_empty()}")
    print(f"   Najlepszy loss: {mycelium.get_best_loss()}")
    
    # Generowanie wag
    shapes = {
        'W1': (2, 4),
        'b1': (1, 4),
        'W2': (4, 1),
        'b2': (1, 1)
    }
    
    print(f"\n2. Pobieranie fusion weights (grzybnia pusta -> losowe):")
    weights1 = mycelium.get_fusion_weights(shapes, alpha=0.7)
    print(f"   W1 shape: {weights1['W1'].shape}")
    print(f"   W1 sample: {weights1['W1'][0, :2]}")
    
    # Symulacja pierwszego agenta
    print(f"\n3. Symulacja agenta #1 (loss=0.5):")
    updated = mycelium.update_memory(weights1, loss=0.5, metadata={"agent_id": "agent_1"})
    print(f"   Zaktualizowano: {updated}")
    print(f"   Najlepszy loss: {mycelium.get_best_loss()}")
    
    # Symulacja drugiego agenta (gorszy wynik)
    print(f"\n4. Symulacja agenta #2 (loss=0.8, gorszy):")
    weights2 = mycelium.get_fusion_weights(shapes, alpha=0.7)
    updated = mycelium.update_memory(weights2, loss=0.8, metadata={"agent_id": "agent_2"})
    print(f"   Zaktualizowano: {updated}")
    print(f"   Najlepszy loss: {mycelium.get_best_loss()}")
    
    # Symulacja trzeciego agenta (lepszy wynik)
    print(f"\n5. Symulacja agenta #3 (loss=0.2, lepszy):")
    weights3 = mycelium.get_fusion_weights(shapes, alpha=0.7)
    updated = mycelium.update_memory(weights3, loss=0.2, metadata={"agent_id": "agent_3"})
    print(f"   Zaktualizowano: {updated}")
    print(f"   Najlepszy loss: {mycelium.get_best_loss()}")
    
    # Statystyki
    print(f"\n6. Statystyki grzybni:")
    stats = mycelium.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Test zakończony!")
