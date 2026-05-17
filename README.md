# NLP - leksykalne upraszczanie

Projekt na zaliczenie przedmiotu "Przetwarzanie Języka Naturalnego" -  porównanie trzech podejść do leksykalnego upraszczania tekstu na zbiorze BenchLS.

## Rozwiązania A i B (Jupyter Notebook)

Rozwiązania oparte na WordNet (`NLP_Proj_WordNet.ipynb`) oraz BERT (`NLP_Proj_BERT.ipynb`) zostały zrealizowane w Google Colab. Wystarczy otworzyć odpowiedni notebook i uruchomić wszystkie komórki po kolei.

---

## Rozwiązanie C — lokalne LLM przez Ollama

Skrypt `NLP_Proj_LLM.py` testuje lokalne modele językowe w trzech wariantach promptowania (Zero-Shot, Few-Shot, Chain-of-Thought) i mierzy skuteczność metryką **Precision@1**.

### Wymagania

- Zainstalowana i uruchomiona aplikacja **[Ollama](https://ollama.com)**
- Pobrany co najmniej jeden model, np.:
```bash
ollama pull llama3.1:8b
```

### Instalacja

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install openai pandas tqdm
```

### Dane

Skrypt oczekuje pliku `benchls.txt` w standardowym formacie tabulatorowym BenchLS:
```
zdanie<TAB>słowo<TAB>id<TAB>zamienniki...
```

### Uruchomienie

```bash
# Domyślnie: model llama3.1:8b, wariant v1
python NLP_Proj_LLM.py

# Kilka modeli naraz
python NLP_Proj_LLM.py --models llama3.1:8b mistral:7b

# Inny wariant promptów
python NLP_Proj_LLM.py --variant v3

# Opóźnienie między zapytaniami (przydatne przy słabszym sprzęcie)
python NLP_Proj_LLM.py --delay 0.5
```

### Argumenty

| Argument | Domyślnie | Opis |
|---|---|---|
| `--benchls` | `benchls.txt` | Ścieżka do pliku ze zbiorem danych |
| `--models` | `llama3.1:8b` | Model lub lista modeli (oddzielone spacją) |
| `--variant` | `v1` | Wariant promptów: `v1`, `v2` lub `v3` |
| `--delay` | `0.0` | Opóźnienie w sekundach między zapytaniami |

### Wyniki

Skrypt wyświetla pasek postępu z aktualną wartością P@1 oraz zapisuje szczegółowy log (np. `eval_llama3.1_8b_20260517_184200.log`) z pełnymi promptami i odpowiedziami modelu.