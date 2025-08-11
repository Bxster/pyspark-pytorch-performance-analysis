# =============================================================================
# train_standalone.py
#
# Esegue l'addestramento e la valutazione di un modello LSTM su dati azionari
# in un ambiente a processo singolo (standalone), senza early stopping.
# Addestramento completo per 200 epoche.
# =============================================================================

# SEZIONE 1: SETUP E IMPORT
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import matplotlib.pyplot as plt
import json
from datetime import datetime
import math

# SEZIONE 2: CONFIGURAZIONE
class Config:
    # --- Percorsi ---
    DATA_DIR = "./"       # Directory dei dati
    OUTPUT_DIR = "./"     # Dove salvare modelli, grafici e risultati

    # --- Dataset ---
    STOCKS_FILE = "sp500_stocks.csv"
    
    # --- Azioni da analizzare ---
    STOCKS_TO_ANALYZE = ['AES', 'ALL', 'CCL', 'GIS']
    
    # --- Feature e target ---
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
    TARGET = 'Close'
    SEQUENCE_LENGTH = 60  # Numero di giorni nella finestra temporale
    
    # --- Parametri di addestramento ---
    TRAIN_SPLIT_RATIO = 0.8
    VALIDATION_SPLIT_RATIO = 0.1
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.001
    
    # --- Parametri del modello ---
    INPUT_SIZE = len(FEATURES)
    HIDDEN_SIZE = 50
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1

# SEZIONE 3: CLASSE DATASET
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# SEZIONE 4: MODELLO LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Prendiamo solo l'ultimo timestep
        out = self.fc(out)
        return out

# SEZIONE 5: METRICHE DI VALUTAZIONE
def calculate_metrics(y_true, y_pred):
    non_zero_mask = y_true != 0    # Evita divisione per zero nel calcolo MAPE
    y_true_safe = y_true[non_zero_mask]
    y_pred_safe = y_pred[non_zero_mask]
    
    mape = np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100 if len(y_true_safe) > 0 else 0
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# FUNZIONE PRINCIPALE
def main():
    config = Config()
    
    # Device: GPU se disponibile, altrimenti CPU (nostro caso)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Analisi su {len(config.STOCKS_TO_ANALYZE)} azioni: {config.STOCKS_TO_ANALYZE}")

    # --- CARICAMENTO E PREPROCESSING DATI ---
    print(f"Caricamento del dataset '{config.STOCKS_FILE}'...")
    stocks_path = os.path.join(config.DATA_DIR, config.STOCKS_FILE)
    try:
        full_stocks_df = pd.read_csv(stocks_path)
        print("Dataset caricato con successo.")
    except FileNotFoundError:
        print(f"ERRORE: File non trovato in '{stocks_path}'.")
        return

    # Filtra tickers
    print(f"Filtraggio per tickers: {config.STOCKS_TO_ANALYZE}")
    data = full_stocks_df[full_stocks_df['Symbol'].isin(config.STOCKS_TO_ANALYZE)].copy()

    # Ordina per data e ticker
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by=['Symbol', 'Date'])
    data.set_index('Date', inplace=True)
    data = data[config.FEATURES + ['Symbol']].dropna()

    # Suddivisione train/val/test
    unique_dates = data.index.unique().sort_values()
    train_end_idx = int(len(unique_dates) * config.TRAIN_SPLIT_RATIO)
    val_end_idx = train_end_idx + int(len(unique_dates) * config.VALIDATION_SPLIT_RATIO)
    train_end_date = unique_dates[train_end_idx]
    val_end_date = unique_dates[val_end_idx]

    train_df = data[data.index <= train_end_date]
    val_df = data[(data.index > train_end_date) & (data.index <= val_end_date)]
    test_df = data[data.index > val_end_date]

    # Normalizzazione
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[config.FEATURES])

    def scale_df(df):
        scaled = pd.DataFrame(scaler.transform(df[config.FEATURES]), columns=config.FEATURES, index=df.index)
        scaled['Symbol'] = df['Symbol']
        return scaled

    train_df_scaled = scale_df(train_df)
    val_df_scaled = scale_df(val_df)
    test_df_scaled = scale_df(test_df)

    # Creazione sequenze
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

    # Conversione in tensori
    X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

    # DataLoader
    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)
    
    # --- INIZIALIZZAZIONE MODELLO LSTM---
    model = LSTMModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE
    ).to(device)

    print(f"\nInfo modello: {sum(p.numel() for p in model.parameters()):,} parametri totali.")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # --- LOOP DI TRAINING ---
    print("\nInizio addestramento (senza early stopping)...")
    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # print(f"Epoch [{epoch+1:03d}/{config.NUM_EPOCHS}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        # Log ogni 10 epoche (più la prima)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{config.NUM_EPOCHS}] - "
                  f"Train Loss: {avg_train_loss:.6f} - "
		              f"Val Loss: {avg_val_loss:.6f}")


    training_time = time.time() - start_time
    print(f"\nAddestramento completato in {training_time:.2f} secondi")

    # --- VALUTAZIONE ---
    print("\nValutazione sul test set...")
    model.eval()
    predictions_scaled = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions_scaled.append(outputs.cpu())
    predictions_scaled = torch.cat(predictions_scaled).numpy()

    # Inverse transform per riportare ai valori reali
    dummy_array_preds = np.zeros((len(predictions_scaled), len(config.FEATURES)))
    dummy_array_preds[:, target_idx] = predictions_scaled.flatten()
    predictions_actual = scaler.inverse_transform(dummy_array_preds)[:, target_idx]

    dummy_array_y = np.zeros((len(y_test), len(config.FEATURES)))
    dummy_array_y[:, target_idx] = y_test.numpy().flatten()
    y_test_actual = scaler.inverse_transform(dummy_array_y)[:, target_idx]

    metrics = calculate_metrics(y_test_actual, predictions_actual)
    print("\nMetriche di Performance sul Test Set:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")

    # --- SALVATAGGIO ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_name_tag = f"Standalone_MultiStock_{len(config.STOCKS_TO_ANALYZE)}tickers"
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    MODEL_PATH = os.path.join(config.OUTPUT_DIR, f"modello_{model_name_tag}_{timestamp}.pth")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModello salvato in: {MODEL_PATH}")

    results_summary = {
        'run_type': 'standalone',
        'stocks_analyzed': config.STOCKS_TO_ANALYZE,
        'total_training_samples': len(X_train),
        'training_time_seconds': round(training_time, 2),
        'aggregate_performance_metrics': {k: round(v, 4) for k, v in metrics.items()},
    }
    RESULTS_PATH = os.path.join(config.OUTPUT_DIR, f"risultati_{model_name_tag}_{timestamp}.json")
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"Riepilogo risultati salvato in: {RESULTS_PATH}")

    # --- PLOT LOSS ---
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Curve di Apprendimento (Training vs Validation Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    PLOT_PATH_LOSS = os.path.join(config.OUTPUT_DIR, f"grafico_loss_{model_name_tag}_{timestamp}.png")
    plt.savefig(PLOT_PATH_LOSS)
    print(f"Grafico Loss salvato in: {PLOT_PATH_LOSS}")
    plt.close()

# --- ESECUZIONE ---
if __name__ == "__main__":
    main()
