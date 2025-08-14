# Analisi delle Prestazioni di PySpark e PyTorch per il Deep Learning Distribuito

Questo repository contiene il codice sorgente e i risultati di un progetto focalizzato sull'analisi delle prestazioni del deep learning distribuito in un ambiente CPU multi-core.

## 🎯 Obiettivo

L'obiettivo principale è confrontare i tempi di addestramento e l'accuratezza di una rete neurale ricorrente (LSTM) in due configurazioni:

1.  **Standalone**: Un'implementazione standard con PyTorch su un singolo processo.
2.  **Distribuita**: Un'implementazione distribuita che sfrutta più core della CPU utilizzando `TorchDistributor` di PySpark.

Il progetto mira a quantificare il guadagno in termini di velocità (speedup) e a valutare l'impatto della distribuzione sull'accuratezza del modello.

## 🛠️ Tecnologie Utilizzate

*   **Python 3.x**
*   **PySpark**: Per la parallelizzazione dei dati e l'orchestrazione dell'addestramento distribuito.
*   **PyTorch**: Per la definizione e l'addestramento del modello LSTM.
*   **Pandas** e **NumPy**: Per la manipolazione e la preparazione dei dati.
*   **Scikit-learn**: Per la valutazione delle metriche di performance.

## 📊 Dataset

Per l'addestramento e il test del modello è stato utilizzato il dataset **S&P 500 Stock Data**, contenente dati storici sulle azioni. È possibile scaricarlo da [Kaggle](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks).

## 🚀 Risultati

Di seguito sono riportate le tabelle con i risultati ottenuti durante i test.

### Tabella 1: Confronto dei Tempi Medi di Addestramento e Speedup

| # Processi (Core) | Tempo Medio (s) | Dev. Std. (s) | Speedup (vs 1 Core) |
| :---------------- | :-------------- | :------------ | :------------------ |
| 1 (Standalone)    | 2053.34         | ± 89.55       | 1.00x (Baseline)    |
| 2 (Distribuito)   | 1050.34         | ± 55.60       | **1.95x**           |
| 3 (Distribuito)   | 894.32          | ± 43.46       | **2.29x**           |
| 4 (Distribuito)   | 789.16          | ± 61.12       | **2.60x**           |

**Commento**: L'utilizzo di PySpark con `TorchDistributor` ha portato a una significativa riduzione dei tempi di addestramento. Passando da 1 a 4 core, abbiamo ottenuto uno **speedup di 2.60x**, dimostrando l'efficacia dell'approccio distribuito anche su una singola macchina multi-core.

### Tabella 2: Confronto delle Metriche Medie di Accuratezza sul Test Set

| # Processi (Core) | MAPE Media (%) | Dev. Std. MAPE (%) | RMSE Medio |
| :---------------- | :------------- | :----------------- | :--------- |
| 1 (Standalone)    | 3.54%          | 0.44%              | 8.58       |
| 2 (Distribuito)   | **2.87%**      | 0.36%              | **5.68**   |
| 3 (Distribuito)   | 3.52%          | 0.41%              | 5.58       |
| 4 (Distribuito)   | 3.23%          | 0.52%              | 5.98       |

**Commento**: Sorprendentemente, la configurazione distribuita con 2 core non solo è stata più veloce, ma ha anche ottenuto le migliori performance in termini di accuratezza, con i valori più bassi di MAPE (Mean Absolute Percentage Error) e RMSE (Root Mean Square Error). Questo suggerisce che la parallelizzazione potrebbe avere un effetto regolarizzatore sull'addestramento in questo specifico scenario.

## 💡 Conclusioni

L'integrazione tra PySpark e PyTorch tramite `TorchDistributor` si è dimostrata una soluzione valida per accelerare il training di modelli di deep learning su architetture multi-core. I risultati mostrano non solo un notevole miglioramento delle prestazioni computazionali, but anche un potenziale impatto positivo sull'accuratezza del modello, aprendo la strada a ulteriori investigazioni.

