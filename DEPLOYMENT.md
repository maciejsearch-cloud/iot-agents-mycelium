# 🚀 Deployment Guide - Streamlit Community Cloud

## Krok 1: Przygotowanie repozytorium GitHub

### 1.1 Inicjalizacja Git (jeśli jeszcze nie zrobione)

```bash
cd c:\Users\kacpe\Desktop\Projekty\iot-agents-project
git init
git add .
git commit -m "Initial commit: IoT Agents with Mycelium Memory"
```

### 1.2 Stwórz repozytorium na GitHub

1. Przejdź na https://github.com/new
2. Nazwa: `iot-agents-mycelium` (lub dowolna)
3. Opis: `IoT Agents with Mycelium Memory - Federated Learning System`
4. Publiczne lub prywatne (oba działają z Streamlit Cloud)
5. **NIE** zaznaczaj "Initialize with README" (mamy już pliki)
6. Kliknij "Create repository"

### 1.3 Połącz lokalny projekt z GitHub

GitHub pokaże Ci komendy - użyj tych dla "existing repository":

```bash
git remote add origin https://github.com/TWOJA_NAZWA/iot-agents-mycelium.git
git branch -M main
git push -u origin main
```

**Zamień `TWOJA_NAZWA` na swoją nazwę użytkownika GitHub!**

---

## Krok 2: Deploy na Streamlit Community Cloud

### 2.1 Zaloguj się do Streamlit Cloud

1. Przejdź na https://share.streamlit.io/
2. Zaloguj się przez GitHub
3. Kliknij "New app"

### 2.2 Konfiguracja aplikacji

**Repository:** Wybierz `TWOJA_NAZWA/iot-agents-mycelium`  
**Branch:** `main`  
**Main file path:** `src/dashboard.py`

**Advanced settings (opcjonalnie):**
- Python version: `3.11` (lub `3.12`)

### 2.3 Deploy!

Kliknij "Deploy!" - aplikacja będzie budowana ~2-5 minut.

---

## Krok 3: Troubleshooting

### Problem: "ModuleNotFoundError"
**Rozwiązanie:** Sprawdź czy `requirements.txt` jest w głównym katalogu projektu (nie w `src/`).

### Problem: "FileNotFoundError: data/mycelium_memory.json"
**Rozwiązanie:** Dodaj do `.gitignore` wyjątek:
```
# W .gitignore zmień:
data/*.json
# na:
data/*.json
!data/.gitkeep
```

Następnie:
```bash
touch data/.gitkeep
git add data/.gitkeep
git commit -m "Add data directory"
git push
```

### Problem: Aplikacja działa lokalnie, ale nie na Cloud
**Rozwiązanie:** Sprawdź ścieżki - na Cloud używaj względnych ścieżek:
```python
# Zamiast:
mycelium = MyceliumMemory("../data/mycelium_memory.json")

# Użyj:
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
mycelium = MyceliumMemory(BASE_DIR / "data" / "mycelium_memory.json")
```

---

## Krok 4: Aktualizacje

Po każdej zmianie w kodzie:

```bash
git add .
git commit -m "Opis zmian"
git push
```

Streamlit Cloud automatycznie przebuduje aplikację (auto-deploy).

---

## 🎯 Gotowe!

Twoja aplikacja będzie dostępna pod adresem:
```
https://TWOJA_NAZWA-iot-agents-mycelium-main-src-dashboard-xyz123.streamlit.app
```

**Link możesz udostępnić komukolwiek!** 🚀

---

## 📝 Dodatkowe opcje

### Secrets (jeśli potrzebujesz API keys)

W Streamlit Cloud dashboard:
1. Kliknij "Settings" → "Secrets"
2. Dodaj w formacie TOML:
```toml
API_KEY = "twoj_klucz"
```

W kodzie:
```python
import streamlit as st
api_key = st.secrets["API_KEY"]
```

### Custom domain

W ustawieniach aplikacji możesz dodać własną domenę (wymaga DNS CNAME).

---

## 🆘 Pomoc

- Dokumentacja: https://docs.streamlit.io/streamlit-community-cloud
- Forum: https://discuss.streamlit.io/
- Status: https://streamlitstatus.com/
