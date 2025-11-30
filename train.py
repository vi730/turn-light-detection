"""
Training Script:
Loads the processed dataset and trains the LRCN (CNN + LSTM) model.
"""

import os
import random
import datetime as dt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from utils import create_dataset, plot_metric

SEED = 27
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["turn_left", "turn_right"]
DATASET_DIR = "data/processed"

def create_lrcn_model(sequence_length, image_height, image_width, num_classes):
    """
    Constructs the LRCN model.
    LRCN = TimeDistributed CNN (Spatial) + LSTM (Temporal)
    """
    model = Sequential()
    
    # CNN Part (Spatial Features)
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'),
                            input_shape=(sequence_length, image_height, image_width, 3)))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Flatten()))
    
    # LSTM Part (Temporal Features)
    model.add(LSTM(32))
    
    #Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()
    return model

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset not found at {DATASET_DIR}. Please run preprocess.py first.")
        return

    # 1. Load Dataset
    print("Loading dataset with sliding window augmentation...")
    features, labels, _ = create_dataset(DATASET_DIR, CLASSES_LIST, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)
    
    print(f"Total sequences created: {len(features)}")
    
    if len(features) == 0:
        print("Error: No features extracted. Check your data path and sequence length.")
        return

    # 2. Split Data
    one_hot_labels = to_categorical(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, one_hot_labels, test_size=0.25, shuffle=True, random_state=SEED)
    
    # 3. Build Model
    model = create_lrcn_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, len(CLASSES_LIST))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    
    # 4. Train
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    print("Starting Training...")
    history = model.fit(
        x=X_train, y=y_train, 
        epochs=70, 
        batch_size=4, 
        shuffle=True, 
        validation_split=0.2, 
        callbacks=[early_stopping]
    )
    
    # 5. Evaluate and Save
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    os.makedirs('models', exist_ok=True)
    model.save('models/lrcn_turn_signal_latest.h5')
    print("Model saved to models/lrcn_turn_signal_latest.h5")
    
    # 6. Plot Results
    plot_metric(history, 'loss', 'val_loss', 'Training_Loss')
    plot_metric(history, 'accuracy', 'val_accuracy', 'Training_Accuracy')

if __name__ == '__main__':
    main()