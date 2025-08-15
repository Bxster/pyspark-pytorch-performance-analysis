# Analisi delle Prestazioni di PySpark e PyTorch per il Deep Learning Distribuito

Questo repository contiene il codice sorgente e i risultati di un progetto focalizzato sull'analisi delle prestazioni del deep learning distribuito in un ambiente CPU multi-core.

## 🎯 Obiettivo

L'obiettivo principale è confrontare i tempi di addestramento e l'accuratezza di una rete neurale ricorrente (LSTM) in due configurazioni:

1.  **Standalone**: Un'implementazione standard con PyTorch su un singolo processo (baseline).
2.  **Distribuita**: Un'implementazione parallelizzata su più core della CPU utilizzando `TorchDistributor` di PySpark.

Il progetto mira a quantificare il guadagno in termini di velocità (**speedup**) e a valutare l'impatto della distribuzione sull'accuratezza del modello.

## 🛠️ Tecnologie Utilizzate

*   **Python 3.9**
*   **Apache Spark (PySpark)**: Per la parallelizzazione dei dati e l'orchestrazione dell'addestramento.
*   **PyTorch**: Per la definizione e l'addestramento del modello LSTM.
*   **Apache Hadoop (HDFS)**: Per i test su storage distribuito.
*   **Pandas** e **NumPy**: Per la manipolazione e la preparazione dei dati.
*   **Scikit-learn**: Per la valutazione delle metriche di performance.

## 📊 Dataset

Per l'addestramento e il test del modello è stato utilizzato il dataset **S&P 500 Stock Data**, contenente dati storici sulle azioni. È possibile scaricarlo da [Kaggle](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks).

## 🚀 Risultati Principali: Scalabilità su CPU

Di seguito sono riportate le tabelle con i risultati ottenuti confrontando l'esecuzione su 1, 2, 3 e 4 core della CPU.

### Tabella 1: Confronto dei Tempi Medi di Addestramento e Speedup

| # Processi (Core) | Tempo Medio (s) | Dev. Std. (s) | Speedup (vs 1 Core) |
| :---------------- | :-------------- | :------------ | :------------------ |
| 1 (Standalone)    | 2053.34         | ± 89.55       | 1.00x (Baseline)    |
| 2 (Distribuito)   | 1050.34         | ± 55.60       | **1.95x**           |
| 3 (Distribuito)   | 894.32          | ± 43.46       | **2.29x**           |
| 4 (Distribuito)   | 789.16          | ± 61.12       | **2.60x**           |

**Commento**: L'utilizzo di PySpark con `TorchDistributor` ha portato a una significativa riduzione dei tempi di addestramento. Passando da 1 a 4 core, abbiamo ottenuto uno **speedup di 2.60x**, dimostrando l'efficacia dell'approccio distribuito anche su una singola macchina multi-core.

### Tabella 2: Confronto delle Metriche Medie di Accuratezza

| # Processi (Core) | MAPE Media (%) | Dev. Std. MAPE (%) | RMSE Medio |
| :---------------- | :------------- | :----------------- | :--------- |
| 1 (Standalone)    | 3.54%          | 0.44%              | 8.58       |
| 2 (Distribuito)   | **2.87%**      | 0.36%              | **5.68**   |
| 3 (Distribuito)   | 3.52%          | 0.41%              | 5.58       |
| 4 (Distribuito)   | 3.23%          | 0.52%              | 5.98       |

**Commento**: L'accelerazione ottenuta non ha compromesso la qualità del modello. Le metriche di accuratezza sono rimaste statisticamente equivalenti in tutti i test, con fluttuazioni minime dovute alla natura stocastica dell'addestramento.

##  bonus: Test su Storage Distribuito (HDFS)

È stato condotto un ulteriore test per valutare l'impatto della lettura dei dati da **HDFS** invece che dal file system locale, simulando uno scenario Big Data più realistico.

| Fonte Dati (4 Core)  | Tempo Medio (s) | Differenza vs Locale |
| :------------------- | :-------------- | :------------------- |
| File System Locale   | 789.16          | Baseline             |
| **HDFS (Warm Avg)**  | **749.14**      | **-5.1% (più veloce)** |

**Commento**: Sorprendentemente, una volta superato il "cold start" iniziale, la lettura da HDFS è risultata in media **più veloce del 5.1%**. Questo risultato, probabilmente dovuto ai meccanismi di caching ottimizzati di Hadoop/Java, conferma la piena compatibilità e l'efficienza della soluzione in un ambiente Big Data end-to-end.

## 💡 Conclusioni

L'integrazione tra PySpark e PyTorch tramite `TorchDistributor` si è dimostrata una soluzione valida ed efficace per accelerare il training di modelli di deep learning su architetture multi-core. I risultati mostrano non solo un notevole miglioramento delle prestazioni computazionali, ma anche il mantenimento della qualità del modello, aprendo la strada a ulteriori investigazioni su cluster multi-nodo.
