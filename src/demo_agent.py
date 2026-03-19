#!/usr/bin/env python3
"""
Demo agenta z wizualizacjami i zapisem do plików
"""

import sys
import os
from pathlib import Path

# Dodaj src do path
sys.path.append(str(Path(__file__).parent))

from optimized_agent import OptimizedXORAgent
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def main():
    """Demo zoptymalizowanego agenta"""
    print("🤖 IoT Agents Project - Zoptymalizowany XOR Agent")
    print("=" * 60)
    
    # Parametry treningu
    print("\n⚙️  Konfiguracja treningu:")
    print("-" * 25)
    
    hidden_dim = 4
    learning_rate = 0.5
    epochs = 20000
    
    print(f"Liczba neuronów ukrytych: {hidden_dim}")
    print(f"Współczynnik uczenia: {learning_rate}")
    print(f"Liczba epok: {epochs}")
    print(f"Optymalizacje: Xavier init, Binary Cross-Entropy, Early stopping")
    
    # Tworzenie agenta
    print(f"\n🧠 Tworzenie zoptymalizowanego agenta XOR...")
    agent = OptimizedXORAgent(hidden_dim=hidden_dim, learning_rate=learning_rate)
    
    # Trening
    print(f"\n🏃 Rozpoczynam trening...")
    agent.train(epochs=epochs, save_interval=1000)
    
    # Upewnij się że katalogi istnieją
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)
    
    # Wizualizacje z zapisem
    print(f"\n📊 Generowanie i zapis wizualizacji...")
    agent.visualize_training("../logs/optimized_training_curves.png")
    agent.visualize_network("../logs/optimized_network_architecture.png")
    
    # Zapis danych
    agent.save_training_data("../data/optimized_training_results.json")
    
    print(f"\n✅ Zakończono!")
    print(f"📁 Wygenerowane pliki:")
    print(f"   - ../logs/optimized_training_curves.png - wykresy treningu")
    print(f"   - ../logs/optimized_network_architecture.png - wizualizacja sieci")
    print(f"   - ../data/optimized_training_results.json - dane treningowe")
    
    # Podsumowanie wyników
    if agent.accuracy_history:
        final_accuracy = agent.accuracy_history[-1]
        final_loss = agent.loss_history[-1]
        print(f"\n📈 Podsumowanie:")
        print(f"   - Finalna dokładność: {final_accuracy:.3f}")
        print(f"   - Finalna strata: {final_loss:.6f}")
        print(f"   - Liczba zapisanych punktów: {len(agent.loss_history)}")
        
        if final_accuracy == 1.0:
            print(f"   - Status: ✅ Perfekcyjne nauczenie XOR!")
        elif final_accuracy >= 0.95:
            print(f"   - Status: ✅ Bardzo dobre wyniki!")
        elif final_accuracy >= 0.75:
            print(f"   - Status: ⚠️  Dobre wyniki")
        else:
            print(f"   - Status: ❌ Wymaga dalszej optymalizacji")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Trening przerwany przez użytkownika")
    except Exception as e:
        print(f"\n❌ Błąd: {e}")
        sys.exit(1)
