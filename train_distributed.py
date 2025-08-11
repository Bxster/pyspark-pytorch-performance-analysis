# =============================================================================
# train_distributed.py
#
# Esegue l'addestramento e la valutazione di un modello LSTM su dati azionari
# in un ambiente distribuito (PySpark + TorchDistributor) su CPU.
# =============================================================================

# SEZIONE 1: SETUP E IMPORT
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import json
from datetime import datetime
import math
import matplotlib.pyplot as plt

# Import per PySpark e TorchDistributor
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

# NUOVA E CORRETTA RIGA DI IMPORT
from pyspark.ml.torch.distributor import TorchDistributor

# SEZIONE 2: CONFIGURAZIONE
class Config:
    # --- Impostazioni dei Percorsi ---
    DATA_DIR = "./" 
    OUTPUT_DIR = "./" # Dove salvare i risultati

    # --- Parametri del Dataset ---
    STOCKS_FILE = "sp500_stocks.csv"
    STOCKS_TO_ANALYZE = ['AES','ALL','CCL','GIS'] # Le stesse 4 azioni della baseline
    
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
    TARGET = 'Close'
    SEQUENCE_LENGTH = 60
    
    # --- Parametri di addestramento ---
    TRAIN_SPLIT_RATIO = 0.8
    VALIDATION_SPLIT_RATIO = 0.1
    BATCH_SIZE = 64 # Mantieni lo stesso batch size per un confronto equo
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.001
    
    # --- Parametri del modello ---
    INPUT_SIZE = len(FEATURES)
    HIDDEN_SIZE = 50
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1

# SEZIONE 4: CLASSE DATASET (INVARIATA)
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# SEZIONE 5: MODELLO LSTM (INVARIATA)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # h0 e c0 devono essere inizializzati per ogni batch (o per la dimensione del batch)
        # e sul device corretto
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # Prende l'output dell'ultimo time step
        out = self.fc(out)
        return out

# SEZIONE 6: METRICHE DI VALUTAZIONE
def calculate_metrics(y_true, y_pred):
    non_zero_mask = y_true != 0
    y_true_safe = y_true[non_zero_mask]
    y_pred_safe = y_pred[non_zero_mask]
    
    mape = np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100 if len(y_true_safe) > 0 else 0
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# =============================================================================
# FUNZIONE DI ADDESTRAMENTO DISTRIBUITO (CHIAMATA DA TORCHDISTRIBUTOR)
# =============================================================================

def train_loop_fn(config_dict):
    # --- Passo 1: Ottieni rank e world_size dalle variabili d'ambiente ---
    # TorchDistributor imposta automaticamente queste variabili per noi.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Converte il dizionario config in un oggetto Config per coerenza
    config = Config()
    for k, v in config_dict.items():
        setattr(config, k, v)

    # Device: GPU se disponibile, altrimenti CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inizializza il processo distribuito
    # backend='gloo' è per la comunicazione su CPU, 'nccl' è per GPU
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    
    # SE IL TUO CODICE VA IN BLOCCO QUI, potrebbe essere necessario impostare l'indirizzo master.
    # In quel caso, aggiungi queste righe prima di init_process_group:
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "29500" # Porta a caso, di solito libera

    print(f"Worker {rank}/{world_size} - Using device: {device}")

    # --- CARICAMENTO E PREPROCESSING DATI (OGNI WORKER CARICA IL FILE) ---
    stocks_path = os.path.join(config.DATA_DIR, config.STOCKS_FILE)
    full_stocks_df = pd.read_csv(stocks_path)
    data = full_stocks_df[full_stocks_df['Symbol'].isin(config.STOCKS_TO_ANALYZE)].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by=['Symbol', 'Date'])
    data.set_index('Date', inplace=True)
    data = data[config.FEATURES + ['Symbol']].dropna()

    unique_dates = data.index.unique().sort_values()
    train_end_idx = int(len(unique_dates) * config.TRAIN_SPLIT_RATIO)
    val_end_idx = train_end_idx + int(len(unique_dates) * config.VALIDATION_SPLIT_RATIO)
    train_end_date = unique_dates[train_end_idx]
    val_end_date = unique_dates[val_end_idx]

    train_df = data[data.index <= train_end_date]
    val_df = data[(data.index > train_end_date) & (data.index <= val_end_date)]
    test_df = data[data.index > val_end_date]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[config.FEATURES])

    train_df_scaled = pd.DataFrame(scaler.transform(train_df[config.FEATURES]), columns=config.FEATURES, index=train_df.index)
    train_df_scaled['Symbol'] = train_df['Symbol']
    val_df_scaled = pd.DataFrame(scaler.transform(val_df[config.FEATURES]), columns=config.FEATURES, index=val_df.index)
    val_df_scaled['Symbol'] = val_df['Symbol']
    test_df_scaled = pd.DataFrame(scaler.transform(test_df[config.FEATURES]), columns=config.FEATURES, index=test_df.index)
    test_df_scaled['Symbol'] = test_df['Symbol']

    def create_sequences_from_scaled(df, seq_length, target_idx):
        all_x, all_y = [], []
        for _, group in df.groupby('Symbol'):
            group_values = group[config.FEATURES].values
            if len(group_values) > seq_length:
                for i in range(len(group_values) - seq_length):
                    all_x.append(group_values[i:(i + seq_length)])
                    all_y.append(group_values[i + seq_length, target_idx])
        return np.array(all_x), np.array(all_y).reshape(-1, 1)

    target_idx = config.FEATURES.index(config.TARGET)
    X_train, y_train = create_sequences_from_scaled(train_df_scaled, config.SEQUENCE_LENGTH, target_idx)
    X_val, y_val = create_sequences_from_scaled(val_df_scaled, config.SEQUENCE_LENGTH, target_idx)
    X_test, y_test = create_sequences_from_scaled(test_df_scaled, config.SEQUENCE_LENGTH, target_idx)
    
    X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

    train_dataset = StockDataset(X_train, y_train)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler)
    
    val_dataset = StockDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    test_dataset = StockDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = LSTMModel(
        input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS, output_size=config.OUTPUT_SIZE
    ).to(device)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=None)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if rank == 0:
        print(f"Inizio addestramento su {world_size} processi...")
    
    worker_start_time = time.time()
    stopped_epoch = config.NUM_EPOCHS
    train_losses, val_losses = [], []
    
    for epoch in range(config.NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # --- Sincronizzazione e Validazione (solo su rank 0) ---
        avg_train_loss_tensor = torch.tensor(train_loss / len(train_loader)).to(device)
        torch.distributed.all_reduce(avg_train_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        avg_train_loss = avg_train_loss_tensor.item() / world_size

        stop_training = False
        if rank == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            avg_val_loss = val_loss / len(val_loader.dataset)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # print(f'Epoch [{epoch+1:03d}/{config.NUM_EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            # Log ogni 10 epoche (più la prima)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:03d}/{config.NUM_EPOCHS}] - "
	            f"Train Loss: {avg_train_loss:.6f} - "
	            f"Val Loss: {avg_val_loss:.6f}")            
            
        
        stop_training_tensor = torch.tensor(int(stop_training), dtype=torch.int).to(device)
        torch.distributed.broadcast(stop_training_tensor, src=0)
        
        if bool(stop_training_tensor.item()):
            break

    worker_training_time = time.time() - worker_start_time
    print(f"Worker {rank}/{world_size} - Addestramento completato in {worker_training_time:.2f} secondi")

    # La valutazione finale e il salvataggio li fa solo il rank 0
    if rank == 0:
        # ... (tutta la logica di valutazione e salvataggio rimane qui, identica a prima)
        # ... (da "print("\nInizio valutazione...") fino a plt.close())
        print("\nInizio valutazione sul test set aggregato (Rank 0)...")
        model.eval()
        predictions_scaled = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                predictions_scaled.append(outputs.cpu())
        predictions_scaled = torch.cat(predictions_scaled).numpy()

        dummy_array_preds = np.zeros((len(predictions_scaled), len(config.FEATURES)))
        dummy_array_preds[:, target_idx] = predictions_scaled.flatten()
        predictions_actual = scaler.inverse_transform(dummy_array_preds)[:, target_idx]

        dummy_array_y = np.zeros((len(y_test), len(config.FEATURES)))
        dummy_array_y[:, target_idx] = y_test.numpy().flatten()
        y_test_actual = scaler.inverse_transform(dummy_array_y)[:, target_idx]

        metrics = calculate_metrics(y_test_actual, predictions_actual)
        print("\nMetriche di Performance Aggregate sul Test Set (Rank 0):")
        for key, value in metrics.items():
            print(f"- {key}: {value:.4f}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        model_name_tag = f"Distributed_MultiStock_{world_size}proc"
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        MODEL_PATH = os.path.join(config.OUTPUT_DIR, f"modello_{model_name_tag}_{timestamp}.pth")
        torch.save(model.module.state_dict(), MODEL_PATH)
        print(f"\nModello salvato in: {MODEL_PATH}")

        results_summary = {
            'run_type': 'distributed', 'num_processes': world_size,
            'stocks_analyzed': config.STOCKS_TO_ANALYZE,
            'total_training_samples': len(X_train),
            'training_time_seconds': round(worker_training_time, 2),
            'stopped_at_epoch': stopped_epoch,
            'aggregate_performance_metrics': {k: round(v, 4) for k, v in metrics.items()},
        }
        RESULTS_PATH = os.path.join(config.OUTPUT_DIR, f"risultati_{model_name_tag}_{timestamp}.json")
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results_summary, f, indent=4)
        print(f"Riepilogo risultati salvato in: {RESULTS_PATH}")

        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Curve di Apprendimento ({world_size} processi)')
        plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)'); plt.legend(); plt.grid(True)
        PLOT_PATH_LOSS = os.path.join(config.OUTPUT_DIR, f"grafico_loss_{model_name_tag}_{timestamp}.png")
        plt.savefig(PLOT_PATH_LOSS)
        print(f"Grafico Loss salvato in: {PLOT_PATH_LOSS}")
        plt.close()

    torch.distributed.destroy_process_group()

# =============================================================================
# FUNZIONE MAIN PER LANCIARE SPARK E TORCHDISTRIBUTOR
# =============================================================================
def main_distributed():
    config = Config() # Crea un'istanza della configurazione

    print(f"Inizio esecuzione distribuita per {len(config.STOCKS_TO_ANALYZE)} azioni con {Config.NUM_PROCESSES} processi...")

    # Inizializza SparkSession. 
    # master="local[*]" usa tutti i core disponibili.
    # Qui useremo "local[NUM_PROCESSES]" per specificare i core.
    # spark.executor.cores=1 è importante per la modalità local[N] per PySpark,
    # altrimenti Spark può tentare di allocare più core per executor.
    spark = SparkSession.builder \
        .appName("DistributedLSTMStockPrediction") \
        .master(f"local[{Config.NUM_PROCESSES}]") \
        .config("spark.executor.cores", "1") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    print("SparkSession inizializzata.")

    # Inizializza TorchDistributor
    distributor = TorchDistributor(
        num_processes=Config.NUM_PROCESSES,
        use_gpu=False, # Imposta a False perché non hai GPU
        local_mode=True # Imposta a True per l'esecuzione su una singola VM
    )

    # Lancia la funzione di addestramento distribuita
    # Passiamo config.__dict__ perché è più semplice serializzare un dizionario
    # e deserializzarlo nella funzione workers.
    results = distributor.run(train_loop_fn, config.__dict__)
    
    # I risultati vengono restituiti dal rank 0 (se impostato così nella train_loop_fn)
    # oppure puoi raccogliere risultati da tutti i worker e aggregarli qui.
    # Nel nostro caso, il rank 0 salva i risultati, quindi non c'è molto da fare qui se non chiudere Spark.
    
    spark.stop()
    print("SparkSession terminata. Esecuzione distribuita completata.")

# =============================================================================
# ESECUZIONE DELLO SCRIPT TRAMITE SPARK-SUBMIT
# =============================================================================

# --- PASSO 1: Inizializza la SparkSession ---
# Quando si usa spark-submit, questo comando si "aggancia" alla sessione
# che spark-submit ha già preparato in background.
spark = SparkSession.builder.appName("DistributedLSTMStockPrediction").getOrCreate()

print("SparkSession recuperata o creata con successo.")
print(f"Versione di Spark: {spark.version}")

# Leggi il numero di processi da una variabile d'ambiente o usa un default.
num_processes = int(os.getenv("NUM_PROCESSES", "2")) 

config = Config()

# --- PASSO 2: Inizializza TorchDistributor ---
distributor = TorchDistributor(
    num_processes=num_processes, 
    local_mode=True, 
    use_gpu=False
)

# --- PASSO 3: Lancia l'addestramento ---
distributor.run(train_loop_fn, config.__dict__)

# --- PASSO 4: Ferma la SparkSession ---
spark.stop()

print(f"\nEsecuzione distribuita con {num_processes} processi completata.")
