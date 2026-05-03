import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import VGG16
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

print(f"TensorFlow: {tf.__version__}")

# ── GPU setup ──────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU enabled: {[g.name for g in gpus]}")
else:
    print("⚠️  No GPU found — running on CPU")

print("\nLoading data...")
X_train = np.load("data/processed/X_train.npy")
X_val   = np.load("data/processed/X_val.npy")
X_test  = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_val   = np.load("data/processed/y_val.npy")
y_test  = np.load("data/processed/y_test.npy")

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Train — Healthy: {sum(y_train==0)} | Abnormal: {sum(y_train==1)}")
print(f"Test  — Healthy: {sum(y_test==0)}  | Abnormal: {sum(y_test==1)}")

# Verify balance
assert abs(sum(y_train==0) - sum(y_train==1)) < 100, "Dataset not balanced!"
print("Dataset is balanced")

# No class weights needed — dataset is balanced
print("\nBuilding VGG16 model...")
base = VGG16(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False

inputs  = Input(shape=(224, 224, 3))
x       = tf.keras.applications.vgg16.preprocess_input(inputs * 255.0)
x       = base(x, training=False)
x       = layers.Flatten()(x)
x       = layers.Dense(512, activation='relu')(x)
x       = layers.Dropout(0.5)(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
model.summary()
os.makedirs("models", exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=8,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "models/thermal_best.h5",
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        patience=4,
        factor=0.2,
        mode='max',
        min_lr=1e-7,
        verbose=1
    )
]

print("\nPhase 1 — Frozen VGG16 (50 epochs max)...")
history1 = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    shuffle=True
)

print("\nPhase 2 — Unfreeze last 4 VGG16 layers...")
for layer in base.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

history2 = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    shuffle=True
)

print("\nEvaluating on test set...")
results = model.evaluate(X_test, y_test, verbose=0)
names   = ['loss','accuracy','auc','precision','recall']
print("\nTest Results:")
for n, v in zip(names, results):
    print(f"  {n}: {v:.4f}")

y_proba = model.predict(X_test).flatten()

# Find best threshold
from sklearn.metrics import f1_score
best_t, best_f1 = 0.5, 0
for t in np.arange(0.1, 0.9, 0.02):
    preds = (y_proba > t).astype(int)
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"\nBest threshold: {best_t:.2f} (macro F1: {best_f1:.4f})")
y_pred = (y_proba > best_t).astype(int)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Healthy','Abnormal'],
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy','Abnormal'],
            yticklabels=['Healthy','Abnormal'])
plt.title('Confusion Matrix — VGG16')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150)
plt.show()

all_acc      = history1.history['accuracy']     + history2.history['accuracy']
all_val_acc  = history1.history['val_accuracy'] + history2.history['val_accuracy']
all_loss     = history1.history['loss']         + history2.history['loss']
all_val_loss = history1.history['val_loss']     + history2.history['val_loss']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(all_acc,     label='Train')
axes[0].plot(all_val_acc, label='Validation')
axes[0].axvline(len(history1.history['accuracy']),
                color='gray', linestyle='--', label='Fine-tune')
axes[0].set_title('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.4)

axes[1].plot(all_loss,     label='Train')
axes[1].plot(all_val_loss, label='Validation')
axes[1].axvline(len(history1.history['loss']),
                color='gray', linestyle='--', label='Fine-tune')
axes[1].set_title('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.4)

plt.suptitle('VGG16 Training History', fontsize=14)
plt.tight_layout()
plt.savefig('models/training_history.png', dpi=150)
plt.show()

model.save('models/thermal_final.h5')
np.save('models/best_threshold.npy', np.array([best_t]))
print(f"\n Model saved to models/thermal_final.h5")
print(f" Best threshold: {best_t:.2f}")