import numpy as np
import os
import glob
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

actions = ['greetings', 'hello', 'meet', 'part', 'glad', 'worry', 'introduction', 'name', 'age', 'you', 'me', 'live', 'know', 'dont know', 'right', 'no', 'what', 'thanks', 'fine', 'want']

data = []
labels = []
no_files = 0

for action in actions:
    data_files = glob.glob(f'dataset/seq_{action}_*.npy')
    no_files += len(data_files)
    for file_path in data_files:
        action_data = np.load(file_path)
        data.append(action_data)
        labels.append(np.array([actions.index(action)] * len(action_data)))

data = np.concatenate(data, axis=0)
labels = to_categorical(np.concatenate(labels, axis=0), num_classes=len(actions))

x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.05, random_state=42)

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)),
    Dropout(0.2),
    BatchNormalization(),
    LSTM(128, return_sequences=False, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(actions.shape[0], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_categorical_accuracy', patience=5, verbose=1),
    ModelCheckpoint('models/best_model.h5', monitor='val_categorical_accuracy', verbose=1, save_best_only=True)
]

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=callbacks)

plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss by Epoch')
plt.xlabel('Epoch')
plt.ylabel('Categorical Crossentropy')
plt.show()

model = load_model('models/best_model.h5')

y_pred = model.predict(x_val)
y_true = np.argmax(y_val, axis=1)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=actions, yticklabels=actions)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(classification_report(y_true, y_pred, target_names=actions))
