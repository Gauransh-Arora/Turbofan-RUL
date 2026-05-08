# =============================================================================
# Turbofan RUL Prediction — V4
# BiLSTM + MC Dropout Uncertainty + Domain Adaptation + Anti-Overfitting
# NASA CMAPSS FD001+FD002+FD003+FD004
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Concatenate,
    BatchNormalization, Bidirectional, Embedding, Flatten
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASETS     = ['FD001', 'FD002', 'FD003', 'FD004']
DATA_PATH    = 'data/'
SEQUENCE_LEN = 30
MAX_RUL      = 150
BATCH_SIZE   = 128
EPOCHS       = 150
MC_SAMPLES = 5          # Monte Carlo forward passes for uncertainty
UNCERTAINTY_THRESHOLD = 10.0  # cycles — flag engines above this

COLUMNS = (
    ['unit', 'cycle'] +
    [f'os_{i}' for i in range(1, 4)] +
    [f's_{i}'  for i in range(1, 22)]
)
DROP_SENSORS = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
OS_COLS      = ['os_1', 'os_2', 'os_3']   # operating condition columns

ZONE_BINS   = [0, 30, 60, 90, 150]
ZONE_LABELS = ['Critical\n(0-30)', 'Warning\n(31-60)',
               'Moderate\n(61-90)', 'Healthy\n(91+)']

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("=" * 60)
print("STEP 1: Loading All 4 Datasets")
print("=" * 60)

def load_txt(path):
    return pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS)

train_parts, test_parts, test_rul_parts = [], [], []

for i, ds in enumerate(DATASETS):
    tr = load_txt(f'{DATA_PATH}train_{ds}.txt')
    te = load_txt(f'{DATA_PATH}test_{ds}.txt')
    rl = pd.read_csv(f'{DATA_PATH}RUL_{ds}.txt', header=None, names=['RUL'])

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

    print(f"  {ds} → train: {tr.shape[0]:>6} rows, "
          f"test: {te.shape[0]:>6} rows, "
          f"engines: {tr['unit'].nunique()}")

train_df  = pd.concat(train_parts,    ignore_index=True)
test_df   = pd.concat(test_parts,     ignore_index=True)
test_last = pd.concat(test_rul_parts, ignore_index=True)

print(f"\nCombined train : {train_df.shape[0]:,} rows, "
      f"{train_df['unit'].nunique()} engines")
print(f"Combined test  : {test_last.shape[0]} engines")

# =============================================================================
# STEP 2: RUL LABELS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Computing RUL Labels")
print("=" * 60)

def add_rul(df, max_rul=MAX_RUL):
    life = df.groupby('unit')['cycle'].max().reset_index()
    life.columns = ['unit', 'max_cycle']
    df = df.merge(life, on='unit')
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=max_rul)
    df.drop(columns=['max_cycle'], inplace=True)
    return df

train_df         = add_rul(train_df)
test_last['RUL'] = test_last['RUL'].clip(upper=MAX_RUL)

print(f"Train RUL — min: {train_df['RUL'].min()}, max: {train_df['RUL'].max()}")
print(f"Test  RUL — min: {test_last['RUL'].min()}, max: {test_last['RUL'].max()}")

# =============================================================================
# STEP 3: DOMAIN ADAPTATION — Operating Condition Clustering
# =============================================================================
# FD001/FD003: 1 condition  → all rows cluster to 1 centroid
# FD002/FD004: 6 conditions → rows spread across 6 centroids
#
# We fit K=6 clusters on operating settings (os_1, os_2, os_3).
# Each row gets a cluster label (0-5) fed as a domain ID to an Embedding layer.
# Sensor values are then normalized WITHIN each cluster — critical fix.
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Domain Adaptation — Operating Condition Clustering")
print("=" * 60)

N_CONDITIONS = 6

kmeans = KMeans(n_clusters=N_CONDITIONS, random_state=42, n_init=20)
kmeans.fit(train_df[OS_COLS])

train_df['condition'] = kmeans.predict(train_df[OS_COLS])
test_df['condition']  = kmeans.predict(test_df[OS_COLS])
test_last['condition'] = test_df.groupby('unit')['condition'].last().values

print(f"Operating condition distribution (train):")
print(train_df['condition'].value_counts().sort_index())

# Per-condition normalization — fit on train, apply to both
feature_cols = [c for c in COLUMNS
                if c not in ['unit', 'cycle'] + DROP_SENSORS + OS_COLS]

scalers = {}
for cond in range(N_CONDITIONS):
    mask = train_df['condition'] == cond
    sc   = MinMaxScaler()

    # Fit only if condition exists in train (FD001/FD003 may miss some)
    if mask.sum() > 0:
        sc.fit(train_df.loc[mask, feature_cols])
        scalers[cond] = sc

# Apply per-condition scaler
def apply_condition_scaler(df, scalers, feature_cols):
    df = df.copy()
    # Cast to float to avoid pandas LossySetitemError when assigning scaled floats to int columns
    df[feature_cols] = df[feature_cols].astype(float)
    for cond, sc in scalers.items():
        mask = df['condition'] == cond
        if mask.sum() > 0:
            df.loc[mask, feature_cols] = sc.transform(
                df.loc[mask, feature_cols]
            )
    return df

train_df = apply_condition_scaler(train_df, scalers, feature_cols)
test_df  = apply_condition_scaler(test_df,  scalers, feature_cols)

print(f"\nUsing {len(feature_cols)} sensor features: {feature_cols}")
print(f"Plus 1 condition ID (embedded) → total inputs: {len(feature_cols)+1}")

# =============================================================================
# STEP 4: SEQUENCE GENERATION (dual input — sensors + condition)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Creating Sequences")
print("=" * 60)

def make_sequences(df, seq_len, feature_cols):
    """Returns X_sensor, X_cond, y"""
    X_s, X_c, y_list = [], [], []
    for _, group in df.groupby('unit'):
        data   = group[feature_cols].values
        conds  = group['condition'].values
        labels = group['RUL'].values
        for i in range(len(data) - seq_len + 1):
            X_s.append(data[i : i + seq_len])
            X_c.append(conds[i + seq_len - 1])   # condition at last timestep
            y_list.append(labels[i + seq_len - 1])
    return (np.array(X_s, dtype=np.float32),
            np.array(X_c, dtype=np.int32),
            np.array(y_list, dtype=np.float32))

def make_test_sequences(df, seq_len, feature_cols):
    """Returns X_sensor, X_cond for test set"""
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

n_units  = train_df['unit'].nunique()
val_ids  = train_df['unit'].unique()[-int(n_units * 0.20):]
val_mask = train_df['unit'].isin(val_ids)

X_train_s, X_train_c, y_train = make_sequences(
    train_df[~val_mask], SEQUENCE_LEN, feature_cols)
X_val_s,   X_val_c,   y_val   = make_sequences(
    train_df[val_mask],  SEQUENCE_LEN, feature_cols)
X_test_s,  X_test_c           = make_test_sequences(
    test_df, SEQUENCE_LEN, feature_cols)
y_test = test_last['RUL'].values.astype(np.float32)

print(f"Train : {X_train_s.shape}, cond: {X_train_c.shape}")
print(f"Val   : {X_val_s.shape},   cond: {X_val_c.shape}")
print(f"Test  : {X_test_s.shape},  cond: {X_test_c.shape}")

# =============================================================================
# STEP 5: ASYMMETRIC HUBER LOSS
# =============================================================================
def asymmetric_huber_loss(y_true, y_pred, delta=10.0, late_penalty=2.0):
    error     = y_pred - y_true
    abs_error = K.abs(error)
    huber     = tf.where(abs_error <= delta,
                         0.5 * K.square(error),
                         delta * (abs_error - 0.5 * delta))
    penalty   = tf.where(error > 0, late_penalty * huber, huber)
    return K.mean(penalty)

# =============================================================================
# STEP 6: MODEL — Dual-input BiLSTM with Domain Embedding
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Building Model")
print("=" * 60)

def build_model(seq_len, n_features, n_conditions):
    l2 = regularizers.l2(3e-4)

    # ── Input A: Sensor time series ──────────────────────────────────────────
    sensor_input = Input(shape=(seq_len, n_features), name='sensor_input')

    x = Bidirectional(
        LSTM(64, return_sequences=True,
             kernel_regularizer=l2,
             recurrent_regularizer=l2,
             recurrent_dropout=0.2),
        name='bilstm1'
    )(sensor_input)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.4, name='drop1')(x)

    x = Bidirectional(
        LSTM(32, return_sequences=False,
             kernel_regularizer=l2,
             recurrent_regularizer=l2,
             recurrent_dropout=0.2),
        name='bilstm2'
    )(x)
    x = BatchNormalization(name='bn2')(x)
    x = Dropout(0.4, name='drop2')(x)

    # ── Input B: Operating condition ID → Embedding ───────────────────────
    # Maps condition 0-5 to a learned 8-dim vector
    # The model learns: "condition 3 means high altitude, weight sensors X/Y"
    cond_input = Input(shape=(1,), dtype='int32', name='condition_input')
    cond_embed = Embedding(
        input_dim=n_conditions,
        output_dim=8,
        name='condition_embedding'
    )(cond_input)
    cond_flat  = Flatten(name='cond_flatten')(cond_embed)

    # ── Merge sensor features + domain embedding ──────────────────────────
    merged = Concatenate(name='merge')([x, cond_flat])

    x = Dense(32, activation='relu', kernel_regularizer=l2, name='dense1')(merged)
    x = Dropout(0.3, name='drop3')(x)
    output = Dense(1, activation='relu', name='rul_output')(x)

    model = Model(
        inputs=[sensor_input, cond_input],
        outputs=output,
        name='BiLSTM_RUL_v4_DomainAdapt'
    )
    return model

model = build_model(SEQUENCE_LEN, len(feature_cols), N_CONDITIONS)
model.summary()

model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss=asymmetric_huber_loss,
    metrics=['mae']
)

# =============================================================================
# STEP 7: COSINE LR + TRAIN
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Training")
print("=" * 60)

def cosine_lr(epoch, initial_lr=3e-4, min_lr=1e-6, total_epochs=150):
    decay = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    return float(min_lr + (initial_lr - min_lr) * decay)

callbacks = [
    EarlyStopping(
        monitor='val_mae', patience=15,
        restore_best_weights=True,
        min_delta=0.1, verbose=1
    ),
    LearningRateScheduler(cosine_lr, verbose=0),
    ModelCheckpoint(
        filepath='best_turbofan_v4.keras',
        monitor='val_mae',
        save_best_only=True, verbose=1
    )
]

model.load_weights("best_turbofan_v4.keras")


# =============================================================================
# STEP 8: MONTE CARLO DROPOUT — Uncertainty Estimation
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: MC Dropout Uncertainty Estimation")
print("=" * 60)

def mc_predict(model, X_sensor, X_cond, n_samples=MC_SAMPLES):
    """
    Keep dropout ACTIVE during inference (training=True).
    Each sample is a stochastic forward pass.
    Returns:
        mean  — best RUL estimate
        std   — epistemic uncertainty (model's confidence)
        all   — (n_samples, n_engines) raw predictions
    """
    preds = np.stack([
        model([X_sensor, X_cond], training=True).numpy().flatten()
        for _ in range(n_samples)
    ], axis=0)
    return preds.mean(axis=0), preds.std(axis=0), preds

y_pred_mean, y_pred_std, y_pred_all = mc_predict(
    model, X_test_s, X_test_c, n_samples=MC_SAMPLES
)

# Flag high-uncertainty engines
flagged_idx  = np.where(y_pred_std > UNCERTAINTY_THRESHOLD)[0]
flagged_df   = pd.DataFrame({
    'engine_idx'     : flagged_idx,
    'actual_RUL'     : y_test[flagged_idx],
    'predicted_RUL'  : y_pred_mean[flagged_idx].round(1),
    'uncertainty_std': y_pred_std[flagged_idx].round(2),
    'dataset'        : test_last['dataset'].values[flagged_idx]
})

print(f"\nEngines flagged (std > {UNCERTAINTY_THRESHOLD} cycles): "
      f"{len(flagged_idx)} / {len(y_test)}")
print(flagged_df.to_string(index=False))

# =============================================================================
# STEP 9: EVALUATE
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Evaluation")
print("=" * 60)

y_pred = y_pred_mean   # use MC mean as final prediction

rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
mae   = mean_absolute_error(y_test, y_pred)
r2    = r2_score(y_test, y_pred)

def nasa_score(y_true, y_pred):
    diff = y_pred - y_true
    return np.sum(np.where(diff < 0,
                           np.exp(-diff / 13) - 1,
                           np.exp( diff / 10) - 1))

score = nasa_score(y_test, y_pred)

print(f"RMSE           : {rmse:.4f} cycles")
print(f"MAE            : {mae:.4f} cycles")
print(f"R² Score       : {r2:.4f}")
print(f"NASA Score     : {score:.2f}  (lower is better)")
print(f"Mean Uncertainty: {y_pred_std.mean():.2f} ± {y_pred_std.std():.2f} cycles")

# =============================================================================

# --- Classification Metrics ---
y_test_bin = (y_test <= 30).astype(int)
y_pred_bin = (y_pred <= 30).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("\n============================================================")
print("BINARY CLASSIFICATION: Is Engine Failing Soon? (RUL <= 30)")
print("============================================================")
print(f"Accuracy : {accuracy_score(y_test_bin, y_pred_bin)*100:.2f}%")
print(f"Precision: {precision_score(y_test_bin, y_pred_bin)*100:.2f}%")
print(f"Recall   : {recall_score(y_test_bin, y_pred_bin)*100:.2f}%")
print(f"F1-Score : {f1_score(y_test_bin, y_pred_bin)*100:.2f}%")

print("\n============================================================")
print("MULTI-CLASS CLASSIFICATION: Health Zones")
print("============================================================")

def to_zones(arr):
    return pd.cut(np.clip(arr, 0, MAX_RUL),
                  bins=ZONE_BINS, labels=ZONE_LABELS,
                  right=True).astype(str)

y_test_zones = to_zones(y_test)
y_pred_zones = to_zones(y_pred)

print(classification_report(y_test_zones, y_pred_zones))
