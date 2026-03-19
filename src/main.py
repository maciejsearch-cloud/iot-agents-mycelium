#!/usr/bin/env python3
"""
Główny plik uruchomieniowy projektu IoT Agents
"""

import sys
import os
from pathlib import Path

# Dodaj src do path
sys.path.append(str(Path(__file__).parent))

from xor_agent import XORAgent
import matplotlib.pyplot as plt


def main():
    """Główna funkcja uruchamiająca agenta"""
    print("🤖 IoT Agents Project - XOR Neural Network")
    print("=" * 50)
    
    # Parametry treningu
    print("\n⚙️  Konfiguracja treningu:")
    print("-" * 20)
    
    hidden_dim = 4
    learning_rate = 0.1
    epochs = 10000
    
    print(f"Liczba neuronów ukrytych: {hidden_dim}")
    print(f"Współczynnik uczenia: {learning_rate}")
    print(f"Liczba epok: {epochs}")
    
    # Tworzenie agenta
    print(f"\n🧠 Tworzenie agenta XOR...")
    agent = XORAgent(hidden_dim=hidden_dim, learning_rate=learning_rate)
    
    # Trening
    print(f"\n🏃 Rozpoczynam trening...")
    agent.train(epochs=epochs, save_interval=1000)
    
    # Wizualizacje
    print(f"\n📊 Generowanie wizualizacji...")
    
    # Upewnij się że katalog data istnieje
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)
    
    # Zapis wizualizacji
    agent.visualize_training("../logs/training_curves.png")
    agent.visualize_network("../logs/network_architecture.png")
    
    # Zapis danych
    agent.save_training_data("../data/training_results.json")
    
    print(f"\n✅ Zakończono!")
    print(f"📁 Sprawdź pliki:")
    print(f"   - ../logs/training_curves.png - wykresy treningu")
    print(f"   - ../logs/network_architecture.png - wizualizacja sieci")
    print(f"   - ../data/training_results.json - dane treningowe")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Trening przerwany przez użytkownika")
    except Exception as e:
        print(f"\n❌ Błąd: {e}")
        sys.exit(1)
