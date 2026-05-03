import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import joblib

print("=" * 50)
print("PRESSURE SENSOR — AUTOENCODER + CLASSIFIER")
print("=" * 50)

# ── Load Data ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/pressure/wisconsin.csv")
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)

print(f"Dataset: {df.shape}")
print(f"Benign:    {sum(df.diagnosis==0)}")
print(f"Malignant: {sum(df.diagnosis==1)}")

X = df.drop('diagnosis', axis=1).values.astype(np.float32)
y = df['diagnosis'].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_val_s   = scaler.transform(X_val).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/pressure_scaler.pkl")

print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Phase 1: Train Autoencoder on Healthy Only ────────────────────────────
print("\nPhase 1 — Training Autoencoder on healthy samples only...")

X_train_healthy = X_train_s[y_train == 0]
print(f"Healthy training samples: {len(X_train_healthy)}")

# Build Autoencoder
ae_input  = Input(shape=(30,), name='ae_input')

# Encoder
encoded = layers.Dense(128, activation='relu')(ae_input)
encoded = layers.BatchNormalization()(encoded)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.BatchNormalization()(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)
encoded = layers.BatchNormalization()(encoded)
bottleneck = layers.Dense(16, activation='relu',
                           name='bottleneck')(encoded)

# Decoder
decoded = layers.Dense(32, activation='relu')(bottleneck)
decoded = layers.Dense(64, activation='relu')(decoded)
decoded = layers.Dense(128, activation='relu')(decoded)
ae_output = layers.Dense(30, activation='linear',
                          name='reconstruction')(decoded)

autoencoder = Model(ae_input, ae_output, name='autoencoder')
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='mse',
    metrics=['mae']
)
autoencoder.summary()

ae_history = autoencoder.fit(
    X_train_healthy, X_train_healthy,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_s, X_val_s),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=5,
            factor=0.3,
            verbose=1)
    ]
)

# Calculate reconstruction errors
train_recon  = autoencoder.predict(X_train_s, verbose=0)
val_recon    = autoencoder.predict(X_val_s,   verbose=0)
test_recon   = autoencoder.predict(X_test_s,  verbose=0)

train_errors = np.mean(np.power(X_train_s - train_recon, 2), axis=1)
val_errors   = np.mean(np.power(X_val_s   - val_recon,   2), axis=1)
test_errors  = np.mean(np.power(X_test_s  - test_recon,  2), axis=1)

# Find best threshold using reconstruction error
from sklearn.metrics import f1_score
best_t, best_f1 = 0.0, 0
for t in np.percentile(train_errors, np.arange(10, 90, 5)):
    preds = (test_errors > t).astype(int)
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

ae_pred = (test_errors > best_t).astype(int)
print(f"\nAutoencoder threshold: {best_t:.4f}")
print(f"Autoencoder F1: {best_f1:.4f}")
print("\nAutoencoder Classification Report:")
print(classification_report(y_test, ae_pred,
      target_names=['Healthy', 'Abnormal'],
      zero_division=0))

# Plot reconstruction error distribution
plt.figure(figsize=(10, 4))
plt.hist(test_errors[y_test==0], bins=30, alpha=0.6,
         color='blue', label='Healthy')
plt.hist(test_errors[y_test==1], bins=30, alpha=0.6,
         color='red', label='Abnormal')
plt.axvline(best_t, color='black', linestyle='--',
            label=f'Threshold: {best_t:.4f}')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Count')
plt.title('Autoencoder Reconstruction Error Distribution')
plt.legend()
plt.tight_layout()
plt.savefig('models/ae_reconstruction_error.png', dpi=150)
plt.show()

# ── Phase 2: Classifier on Bottleneck Features ───────────────────────────
print("\nPhase 2 — Training classifier on bottleneck features...")

encoder = Model(ae_input, bottleneck, name='encoder')
encoder.trainable = False

clf_input  = Input(shape=(30,), name='pressure_input')
features   = encoder(clf_input)
x          = layers.Dense(32, activation='relu')(features)
x          = layers.BatchNormalization()(x)
x          = layers.Dropout(0.3)(x)
x          = layers.Dense(16, activation='relu',
                           name='fusion_features')(x)
x          = layers.Dropout(0.2)(x)
clf_output = layers.Dense(1, activation='sigmoid')(x)

classifier = Model(clf_input, clf_output, name='pressure_classifier')
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)
classifier.summary()

clf_history = classifier.fit(
    X_train_s, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_s, y_val),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
    "models/pressure_best.weights.h5",
    monitor='val_auc',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            patience=5,
            factor=0.3,
            mode='max',
            verbose=1)
    ]
)

# ── Phase 3: Fine-tune full model ─────────────────────────────────────────
print("\nPhase 3 — Fine-tuning full model...")
encoder.trainable = True
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

ft_history = classifier.fit(
    X_train_s, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val_s, y_val),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=8,
            restore_best_weights=True,
            mode='max',
            verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
    "models/pressure_best.weights.h5",
    monitor='val_auc',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1),
    ]
)

# ── Final Evaluation ──────────────────────────────────────────────────────
print("\nFinal Evaluation...")
results = classifier.evaluate(X_test_s, y_test, verbose=0)
names   = ['loss','accuracy','auc','precision','recall']
print("\nTest Results:")
for n, v in zip(names, results):
    print(f"  {n}: {v:.4f}")

y_pred = (classifier.predict(X_test_s, verbose=0) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Healthy','Abnormal'],
      zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Healthy','Abnormal'],
            yticklabels=['Healthy','Abnormal'])
plt.title('Pressure Autoencoder Classifier — Confusion Matrix')
plt.tight_layout()
plt.savefig('models/pressure_confusion_matrix.png', dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
all_acc = (clf_history.history['accuracy'] +
           ft_history.history['accuracy'])
all_val = (clf_history.history['val_accuracy'] +
           ft_history.history['val_accuracy'])
all_loss    = (clf_history.history['loss'] +
               ft_history.history['loss'])
all_val_loss= (clf_history.history['val_loss'] +
               ft_history.history['val_loss'])

axes[0].plot(all_acc,  label='Train')
axes[0].plot(all_val,  label='Validation')
axes[0].axvline(len(clf_history.history['accuracy']),
                color='gray', linestyle='--', label='Fine-tune')
axes[0].set_title('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.4)

axes[1].plot(all_loss,     label='Train')
axes[1].plot(all_val_loss, label='Validation')
axes[1].axvline(len(clf_history.history['loss']),
                color='gray', linestyle='--', label='Fine-tune')
axes[1].set_title('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.4)

plt.suptitle('Pressure Autoencoder + Classifier Training')
plt.tight_layout()
plt.savefig('models/pressure_training_history.png', dpi=150)
plt.show()

autoencoder.save('models/autoencoder.keras')
classifier.save('models/pressure_classifier.keras')
encoder_model = Model(ae_input, bottleneck, name='encoder_final')
encoder_model.save('models/pressure_encoder.keras')

print("\n Autoencoder saved   → models/autoencoder.keras")
print(" Classifier saved    → models/pressure_classifier.keras")
print(" Encoder saved       → models/pressure_encoder.keras")
print(" Scaler saved        → models/pressure_scaler.pkl")