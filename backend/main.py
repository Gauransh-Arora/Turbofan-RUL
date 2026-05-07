import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASETS     = ['FD001', 'FD002', 'FD003', 'FD004']
DATA_PATH    = '../data/'  # assuming running from backend/
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/' # fallback if running from root

SEQUENCE_LEN = 30
MAX_RUL      = 150
MC_SAMPLES   = 50
UNCERTAINTY_THRESHOLD = 10.0

COLUMNS = (
    ['unit', 'cycle'] +
    [f'os_{i}' for i in range(1, 4)] +
    [f's_{i}'  for i in range(1, 22)]
)
DROP_SENSORS = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
OS_COLS      = ['os_1', 'os_2', 'os_3']
N_CONDITIONS = 6
CACHE_DIR    = 'cache'

# Globals for server state
model = None
scalers = {}
kmeans = None
feature_cols = []
test_df = pd.DataFrame()
test_last = pd.DataFrame()
test_rul = pd.DataFrame()

app = FastAPI(title="Aerospace RUL Monitor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_txt(path):
    return pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS)

def asymmetric_huber_loss(y_true, y_pred, delta=10.0, late_penalty=2.0):
    error     = y_pred - y_true
    abs_error = tf.keras.backend.abs(error)
    huber     = tf.where(abs_error <= delta,
                         0.5 * tf.keras.backend.square(error),
                         delta * (abs_error - 0.5 * delta))
    penalty   = tf.where(error > 0, late_penalty * huber, huber)
    return tf.keras.backend.mean(penalty)

@app.on_event("startup")
async def startup_event():
    global model, scalers, kmeans, feature_cols, test_df, test_last, test_rul
    print("Loading data and fitting scalers...")
    
    train_parts, test_parts, test_rul_parts = [], [], []
    for i, ds in enumerate(DATASETS):
        tr = load_txt(os.path.join(DATA_PATH, f'train_{ds}.txt'))
        te = load_txt(os.path.join(DATA_PATH, f'test_{ds}.txt'))
        rl = pd.read_csv(os.path.join(DATA_PATH, f'RUL_{ds}.txt'), header=None, names=['RUL'])
        
        unit_offset   = i * 10000
        tr['unit']   += unit_offset
        te['unit']   += unit_offset
        tr['dataset'] = ds
        te['dataset'] = ds
        
        test_last_ds            = te.groupby('unit').last().reset_index()
        test_last_ds['RUL']     = rl['RUL'].values
        test_last_ds['dataset'] = ds
        
        train_parts.append(tr)
        test_parts.append(te)
        test_rul_parts.append(test_last_ds)
        
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    test_last = pd.concat(test_rul_parts, ignore_index=True)
    
    test_last['RUL'] = test_last['RUL'].clip(upper=MAX_RUL)
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=N_CONDITIONS, random_state=42, n_init=20)
    kmeans.fit(train_df[OS_COLS])
    
    train_df['condition'] = kmeans.predict(train_df[OS_COLS])
    test_df['condition']  = kmeans.predict(test_df[OS_COLS])
    
    global feature_cols
    feature_cols = [c for c in COLUMNS if c not in ['unit', 'cycle'] + DROP_SENSORS + OS_COLS]
    
    # Fit Scalers
    for cond in range(N_CONDITIONS):
        mask = train_df['condition'] == cond
        sc = MinMaxScaler()
        if mask.sum() > 0:
            sc.fit(train_df.loc[mask, feature_cols])
            scalers[cond] = sc
            
    # Scale test_df
    test_df[feature_cols] = test_df[feature_cols].astype(float)
    for cond, sc in scalers.items():
        mask = test_df['condition'] == cond
        if mask.sum() > 0:
            test_df.loc[mask, feature_cols] = sc.transform(test_df.loc[mask, feature_cols])
            
    print("Loading model...")
    model_path = 'turbofan_rul_v4.keras'
    if not os.path.exists(model_path) and os.path.exists('../turbofan_rul_v4.keras'):
        model_path = '../turbofan_rul_v4.keras'
    model = tf.keras.models.load_model(model_path, custom_objects={'asymmetric_huber_loss': asymmetric_huber_loss})
    
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    print("Startup complete.")

def make_test_sequences(df, seq_len, feature_cols):
    X_s, X_c = [], []
    for _, group in df.groupby('unit'):
        data  = group[feature_cols].values
        conds = group['condition'].values
        if len(data) >= seq_len:
            X_s.append(data[-seq_len:])
        else:
            pad = np.zeros((seq_len - len(data), len(feature_cols)))
            X_s.append(np.vstack([pad, data]))
        X_c.append(int(conds[-1]))
    return (np.array(X_s, dtype=np.float32),
            np.array(X_c, dtype=np.int32))

def mc_predict(model, X_sensor, X_cond, n_samples=MC_SAMPLES):
    preds = np.stack([
        model([X_sensor, X_cond], training=True).numpy().flatten()
        for _ in range(n_samples)
    ], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)

def get_health_status(rul):
    if rul <= 30: return 'Critical'
    if rul <= 60: return 'Warning'
    if rul <= 90: return 'Moderate'
    return 'Healthy'

@app.get("/api/engines")
def get_engines():
    cache_path = os.path.join(CACHE_DIR, "fleet_summary.json")
    if os.path.exists(cache_path):
        print("Serving fleet summary from cache...")
        with open(cache_path, 'r') as f:
            return json.load(f)

    # Return a curated subset of engines for the dashboard (e.g., first 5 from each dataset)
    selected_units = []
    for ds in DATASETS:
        units = test_last[test_last['dataset'] == ds]['unit'].head(5).tolist()
        selected_units.extend(units)
        
    subset_last = test_last[test_last['unit'].isin(selected_units)].copy()
    subset_df = test_df[test_df['unit'].isin(selected_units)].copy()
    
    X_test_s, X_test_c = make_test_sequences(subset_df, SEQUENCE_LEN, feature_cols)
    y_pred_mean, y_pred_std = mc_predict(model, X_test_s, X_test_c, n_samples=10) # Less samples for fast listing
    
    results = []
    for i, unit in enumerate(selected_units):
        cycle = int(subset_last[subset_last['unit'] == unit]['cycle'].values[0])
        rul = float(y_pred_mean[i])
        results.append({
            "id": f"ENG-{unit}",
            "original_unit": int(unit),
            "cycle": cycle,
            "rul_predicted": max(0, round(rul)),
            "rul_std": round(float(y_pred_std[i]), 1),
            "status": get_health_status(rul)
        })
    
    with open(cache_path, 'w') as f:
        json.dump(results, f)
        
    return results

@app.get("/api/engines/{unit_id}/telemetry")
def get_engine_telemetry(unit_id: int):
    cache_path = os.path.join(CACHE_DIR, f"engine_{unit_id}.json")
    if os.path.exists(cache_path):
        print(f"Serving engine {unit_id} telemetry from cache...")
        with open(cache_path, 'r') as f:
            return json.load(f)

    # Historical telemetry and degradation predictions
    engine_data = test_df[test_df['unit'] == unit_id].copy()
    if engine_data.empty:
        return {"error": "Engine not found"}
        
    # We will simulate the degradation curve by running prediction at each timestep
    # To keep it fast, we will only predict for the last 50 cycles or all if < 50
    cycles = engine_data['cycle'].values
    
    telemetry = []
    # In test_df, the sensors are scaled. We should unscale them or just return them scaled.
    # T24 is s_2, P30 is s_3. They are at indices 0 and 1 in feature_cols assuming s_1 is dropped.
    # Actually let's just return scaled values for charting, or grab raw from somewhere. 
    # Since we modified test_df in place, it's scaled. We'll return scaled values for T24, P30.
    
    t24_idx = feature_cols.index('s_2') if 's_2' in feature_cols else 0
    p30_idx = feature_cols.index('s_3') if 's_3' in feature_cols else 1
    
    for i, cycle in enumerate(cycles):
        row = engine_data.iloc[i]
        telemetry.append({
            "cycle": int(cycle),
            "t24": float(row[feature_cols[t24_idx]]),
            "p30": float(row[feature_cols[p30_idx]])
        })
        
    # Degradation Curve - let's predict for the last 20 steps to show trajectory
    degradation = []
    start_idx = max(0, len(cycles) - 20)
    
    batch_X_s, batch_X_c = [], []
    for i in range(start_idx, len(cycles)):
        # sequence up to i
        seq_data = engine_data.iloc[:i+1]
        X_s, X_c = make_test_sequences(seq_data, SEQUENCE_LEN, feature_cols)
        batch_X_s.append(X_s[0])
        batch_X_c.append(X_c[0])
        
    if batch_X_s:
        batch_X_s = np.array(batch_X_s, dtype=np.float32)
        batch_X_c = np.array(batch_X_c, dtype=np.int32)
        n_actual = len(batch_X_s)
        
        # Pad to exactly 20 to avoid TF recurrent_dropout static batch size caching issues
        if n_actual < 20:
            pad_s = np.zeros((20 - n_actual, SEQUENCE_LEN, len(feature_cols)), dtype=np.float32)
            pad_c = np.zeros((20 - n_actual,), dtype=np.int32)
            batch_X_s = np.concatenate([batch_X_s, pad_s], axis=0)
            batch_X_c = np.concatenate([batch_X_c, pad_c], axis=0)
            
        mean_rul, std_rul = mc_predict(model, batch_X_s, batch_X_c, n_samples=10)
        
        for idx in range(n_actual):
            m_val = float(mean_rul[idx])
            s_val = float(std_rul[idx])
            degradation.append({
                "cycle": int(cycles[start_idx + idx]),
                "predicted": m_val,
                "upper": m_val + (2 * s_val),
                "lower": m_val - (2 * s_val)
            })
        
    response = {
        "telemetry": telemetry,
        "degradation": degradation,
        "details": {
            "install_date": "2023-01-15",
            "model": "CMAPSS-BiLSTM"
        }
    }
    
    with open(cache_path, 'w') as f:
        json.dump(response, f)
        
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
