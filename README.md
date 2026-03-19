# IoT Agents Project - Mushroom Memory Architecture

## Koncept
Aplikacja IoT w której użytkownicy za kliknięcie raz na X czasu uruchamiają losowego agenta ML, który uczy się i zbiera dane. Agenci mają wspólną pamięć na wzór grzybów (mycelium network).

## Architektura (wagi priorytetów)

### 🔴 PRIORYTET 1 - Core System (waga: 10/10)
- `src/main.py` - główna aplikacja, orchestrator agentów
- `src/agent.py` - klasa bazowa agenta z logiką XOR
- `src/memory.py` - system pamięci wspólnych (grzybowy)
- `src/iot_interface.py` - interfejs dla urządzeń IoT

### 🟡 PRIORYTET 2 - Agent System (waga: 8/10)
- `agents/xor_agent.py` - implementacja agenta uczącego się XOR
- `agents/agent_factory.py` - fabryka tworzenia losowych agentów
- `agents/agent_manager.py` - zarządzanie cyklem życia agentów

### 🟢 PRIORYTET 3 - Dane i Logowanie (waga: 6/10)
- `data/training_data.py` - zarządzanie danymi treningowymi
- `logs/logger.py` - system logowania
- `data/memory_storage.py` - trwały zapis pamięci

### 🔵 PRIORYTET 4 - Testy i UI (waga: 4/10)
- `tests/test_agent.py` - testy jednostkowe agentów
- `tests/test_memory.py` - testy systemu pamięci
- `src/simple_ui.py` - prosty interfejs użytkownika

## Stack Technologiczny
- **Python 3.12+**
- **NumPy** - tylko dla obliczeń ML (zero PyTorch/TF)
- **Struktura modułowa** - łatwość rozbudowy

## Plusy podejścia Python + NumPy

### ✅ Zalety
1. **Pełna kontrola** - widzisz każdą operację matematyczną
2. **Edukacyjne** - idealne do zrozumienia deep learning od podstaw
3. **Szybkość prototypowania** - minimalne zależności
4. **Debugging** - łatwe śledzenie forward/backpropagation
5. **Niski próg wejścia** - wystarczy podstawowa znajomość Python
6. **Wydajność** - NumPy jest zoptymalizowany pod CPU
7. **Przenośność** - działa wszędzie gdzie Python

### ❌ Wady
1. **Brak GPU** - ograniczona skalowalność
2. **Ręczne implementacje** - więcej kodu do napisania
3. **Brak autograd** - ręczne backpropagation
4. **Mniejsze ekosystemy** - mniej gotowych rozwiązań
5. **Ograniczone modele** - trudniej o zaawansowane architektury

## Możliwe ulepszenia

### 🚀 Krótkoterminowe (v1.1)
1. **Wizualizacja nauki** - matplotlib dla loss curves
2. **Zapis/odczyt wag** - persistencja modelu
3. **Konfiguracja JSON** - parametry treningu
4. **Prosty web UI** - Flask/FastAPI
5. **Logi strukturalne** - JSON format

### 🔮 Średnioterminowe (v2.0)
1. **Przejście na PyTorch** - GPU + autograd
2. **Wiele zadań** - nie tylko XOR
3. **Agent hierarchy** - masters/slaves
4. **Pamięć wektorowa** - embeddings
5. **Real IoT devices** - MQTT/HTTP API

### 🌈 Długoterminowe (v3.0)
1. **Transformers** - attention mechanism
2. **Federated learning** - uczenie rozproszone
3. **Reinforcement learning** - nagrody za działania
4. **Edge deployment** - modele na urządzeniach
5. **Explainable AI** - interpretowalność decyzji

## ✅ Zaimplementowane Funkcjonalności

### 🍄 Mycelium Memory (Federated Learning)
- Thread-safe & process-safe pamięć wspólna (FileLock)
- Fusion weights: mieszanie najlepszych wag z grzybni + losowe (alpha blending)
- Automatyczna aktualizacja przy lepszym loss
- Pełne API: stats, metadata, reset

### 🤖 Universal Agent
- Obsługa 8 różnych zadań (XOR, AND, OR, NAND, NOR, XNOR, 3-bit Parity, Majority)
- Automatyczne dostosowanie architektury do wymiaru zadania
- Xavier initialization + Binary Cross-Entropy
- Early stopping + historia treningu

### 🌐 IoT Orchestrator (Asyncio)
- Asynchroniczna symulacja wielu urządzeń IoT
- asyncio.to_thread() dla CPU-bound treningu
- Live monitoring agentów
- Kolektywne uczenie się przez Mycelium

### 📊 Streamlit Dashboard
- **Live monitoring** - statystyki w czasie rzeczywistym
- **Interaktywne wykresy** - Plotly (loss, accuracy)
- **Kontrolki parametrów** - hidden_dim, learning_rate, epochs, alpha
- **8 zadań do wyboru** - od prostych (AND) do trudnych (3-bit Parity)
- **Tabela predykcji** - z kolorowaniem wyników
- **Historia Mycelium** - metadane i statystyki

## 🚀 Quick Start

### Instalacja
```bash
pip install -r requirements.txt
```

### Uruchomienie Dashboard
```bash
python -m streamlit run src/dashboard.py
```
Otwórz http://localhost:8501 w przeglądarce.

### Testowanie zadań
```bash
python src/tasks.py          # Lista wszystkich zadań
python src/universal_agent.py # Test treningu
```

### Symulacja IoT
```bash
python src/iot_orchestrator.py  # 30s symulacji, 5 urządzeń
```

## 📦 Deployment na Streamlit Cloud

Szczegółowe instrukcje w `DEPLOYMENT.md`.

**Szybki start:**
1. Stwórz repo na GitHub
2. Push kodu: `git push origin main`
3. Deploy na https://share.streamlit.io/
4. Wybierz repo i `src/dashboard.py`

## 📁 Struktura Projektu

```
iot-agents-project/
├── src/
│   ├── dashboard.py          # Streamlit Dashboard
│   ├── memory.py             # Mycelium Memory (thread-safe)
│   ├── tasks.py              # 8 zadań uczenia
│   ├── universal_agent.py    # Uniwersalny agent ML
│   ├── optimized_agent.py    # Zoptymalizowany agent XOR
│   └── iot_orchestrator.py   # Asyncio orchestrator
├── data/
│   └── mycelium_memory.json  # Pamięć wspólna agentów
├── logs/                     # Wizualizacje i logi
├── tests/                    # Testy jednostkowe
├── .streamlit/
│   └── config.toml           # Konfiguracja Streamlit
├── requirements.txt          # Zależności
├── README.md                 # Ten plik
└── DEPLOYMENT.md             # Instrukcje wdrożenia
