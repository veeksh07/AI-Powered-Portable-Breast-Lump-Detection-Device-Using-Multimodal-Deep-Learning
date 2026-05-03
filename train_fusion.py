import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import joblib

print("=" * 50)
print("MULTIMODAL FUSION MODEL")
print("=" * 50)
print(f"TensorFlow: {tf.__version__}")

# ── Load thermal data ─────────────────────────────────────────────────────
print("\nLoading thermal data...")
X_thermal_train = np.load("data/processed/X_train.npy")
X_thermal_val   = np.load("data/processed/X_val.npy")
X_thermal_test  = np.load("data/processed/X_test.npy")
y_thermal_train = np.load("data/processed/y_train.npy")
y_thermal_val   = np.load("data/processed/y_val.npy")
y_thermal_test  = np.load("data/processed/y_test.npy")

print(f"Thermal — Train: {len(X_thermal_train)} | Val: {len(X_thermal_val)} | Test: {len(X_thermal_test)}")

# ── Load pressure data ────────────────────────────────────────────────────
print("\nLoading pressure data...")
df = pd.read_csv("data/pressure/wisconsin.csv")
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)

X_p = df.drop('diagnosis', axis=1).values.astype(np.float32)
y_p = df['diagnosis'].values

X_p_train, X_p_temp, y_p_train, y_p_temp = train_test_split(
    X_p, y_p, test_size=0.3, random_state=42, stratify=y_p)
X_p_val, X_p_test, y_p_val, y_p_test = train_test_split(
    X_p_temp, y_p_temp, test_size=0.5, random_state=42, stratify=y_p_temp)

scaler      = joblib.load("models/pressure_scaler.pkl")
X_p_train_s = scaler.transform(X_p_train).astype(np.float32)
X_p_val_s   = scaler.transform(X_p_val).astype(np.float32)
X_p_test_s  = scaler.transform(X_p_test).astype(np.float32)

print(f"Pressure — Train: {len(X_p_train_s)} | Val: {len(X_p_val_s)} | Test: {len(X_p_test_s)}")

# ── Align dataset sizes ───────────────────────────────────────────────────
n_train = min(len(X_thermal_train), len(X_p_train_s))
n_val   = min(len(X_thermal_val),   len(X_p_val_s))
n_test  = min(len(X_thermal_test),  len(X_p_test_s))

X_thermal_train = X_thermal_train[:n_train]
X_p_train_s     = X_p_train_s[:n_train]
y_train         = y_thermal_train[:n_train]

X_thermal_val   = X_thermal_val[:n_val]
X_p_val_s       = X_p_val_s[:n_val]
y_val           = y_thermal_val[:n_val]

X_thermal_test  = X_thermal_test[:n_test]
X_p_test_s      = X_p_test_s[:n_test]
y_test          = y_thermal_test[:n_test]

print(f"\nAligned — Train: {n_train} | Val: {n_val} | Test: {n_test}")

# ── Build Thermal Branch (VGG16) ──────────────────────────────────────────
print("\nBuilding thermal branch (VGG16)...")
vgg = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
vgg.trainable = False

thermal_input = Input(shape=(224, 224, 3), name='thermal_input')
x_t = tf.keras.applications.vgg16.preprocess_input(thermal_input * 255.0)
x_t = vgg(x_t, training=False)
x_t = layers.Flatten()(x_t)
x_t = layers.Dense(512, activation='relu')(x_t)
x_t = layers.Dropout(0.5)(x_t)
thermal_features = layers.Dense(128, activation='relu',
                                  name='thermal_features')(x_t)

# Load thermal weights
print("Loading thermal weights...")
thermal_branch = Model(thermal_input, thermal_features, name='thermal_branch')

# Build full thermal model to load weights
full_thermal_input = Input(shape=(224, 224, 3))
x_full = tf.keras.applications.vgg16.preprocess_input(full_thermal_input * 255.0)
x_full = vgg(x_full, training=False)
x_full = layers.Flatten()(x_full)
x_full = layers.Dense(512, activation='relu')(x_full)
x_full = layers.Dropout(0.5)(x_full)
x_full = layers.Dense(256, activation='relu')(x_full)
x_full = layers.Dropout(0.4)(x_full)
full_output = layers.Dense(1, activation='sigmoid')(x_full)
full_thermal = Model(full_thermal_input, full_output)
full_thermal.load_weights("models/thermal_best.keras", by_name=False, skip_mismatch=True)
print(" Thermal weights loaded")

# ── Build Pressure Branch (Autoencoder Encoder) ───────────────────────────
print("Building pressure branch (autoencoder encoder)...")
pressure_input = Input(shape=(30,), name='pressure_input')
x_p = layers.Dense(128, activation='relu')(pressure_input)
x_p = layers.BatchNormalization()(x_p)
x_p = layers.Dense(64,  activation='relu')(x_p)
x_p = layers.BatchNormalization()(x_p)
x_p = layers.Dense(32,  activation='relu')(x_p)
x_p = layers.BatchNormalization()(x_p)
pressure_features = layers.Dense(16, activation='relu',
                                   name='pressure_features')(x_p)

pressure_branch = Model(pressure_input, pressure_features,
                         name='pressure_branch')
pressure_branch.load_weights("models/pressure_encoder.keras",
                               by_name=False, skip_mismatch=True)
print(" Pressure weights loaded")

# ── Freeze branches ───────────────────────────────────────────────────────
thermal_branch.trainable  = False
pressure_branch.trainable = False

# ── Build Fusion Model ────────────────────────────────────────────────────
print("\nBuilding fusion model...")
t_in = Input(shape=(224, 224, 3), name='thermal_input')
p_in = Input(shape=(30,),         name='pressure_input')

t_feat = thermal_branch(t_in,  training=False)
p_feat = pressure_branch(p_in, training=False)

merged = layers.concatenate([t_feat, p_feat], name='fusion')
x = layers.Dense(256, activation='relu')(merged)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64,  activation='relu')(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(1, activation='sigmoid', name='diagnosis')(x)

fusion_model = Model(inputs=[t_in, p_in], outputs=output,
                      name='multimodal_fusion')
fusion_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)
fusion_model.summary()
os.makedirs("models", exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=8,
        restore_best_weights=True, mode='max', verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        "models/fusion_best.weights.h5",
        monitor='val_auc', save_best_only=True,
        save_weights_only=True, mode='max', verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc', patience=4,
        factor=0.3, mode='max', verbose=1)
]

# ── Phase 1 — Frozen branches ─────────────────────────────────────────────
print("\nPhase 1 — Training fusion layers (frozen branches)...")
history1 = fusion_model.fit(
    {'thermal_input': X_thermal_train,
     'pressure_input': X_p_train_s},
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(
        {'thermal_input': X_thermal_val,
         'pressure_input': X_p_val_s},
        y_val),
    callbacks=callbacks,
    shuffle=True
)

# ── Phase 2 — Fine-tune all ───────────────────────────────────────────────
print("\nPhase 2 — Fine-tuning all layers...")
thermal_branch.trainable  = True
pressure_branch.trainable = True
vgg.trainable = False

fusion_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

history2 = fusion_model.fit(
    {'thermal_input': X_thermal_train,
     'pressure_input': X_p_train_s},
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(
        {'thermal_input': X_thermal_val,
         'pressure_input': X_p_val_s},
        y_val),
    callbacks=callbacks,
    shuffle=True
)

# ── Evaluate ──────────────────────────────────────────────────────────────
print("\nEvaluating fusion model...")
results = fusion_model.evaluate(
    {'thermal_input': X_thermal_test,
     'pressure_input': X_p_test_s},
    y_test, verbose=0)

names = ['loss','accuracy','auc','precision','recall']
print("\nFusion Model Test Results:")
for n, v in zip(names, results):
    print(f"  {n}: {v:.4f}")

y_pred = (fusion_model.predict(
    {'thermal_input': X_thermal_test,
     'pressure_input': X_p_test_s},
    verbose=0) > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Healthy','Abnormal'],
      zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Healthy','Abnormal'],
            yticklabels=['Healthy','Abnormal'])
plt.title('Multimodal Fusion — Confusion Matrix')
plt.tight_layout()
plt.savefig('models/fusion_confusion_matrix.png', dpi=150)
plt.show()

fusion_model.save_weights('models/fusion_final.weights.h5')
print("\n Fusion weights saved → models/fusion_final.weights.h5")
print(" Confusion matrix    → models/fusion_confusion_matrix.png")
print("\n Multimodal fusion model complete!")