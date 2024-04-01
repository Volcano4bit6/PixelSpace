import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
# import mediapipe as mp
import tensorflow as tf
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import (LSTM, Dense, Concatenate, Attention, Dropout, Softmax,
                                     Input, Flatten, Activation, Bidirectional, Permute, multiply, 
                                     ConvLSTM2D, MaxPooling3D, TimeDistributed, Conv2D, MaxPooling2D)

from scipy import stats

# disable some of the tf/keras training warnings 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)

# suppress untraced functions warning
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import csv
import json

def find_max_acc(yhat, res):
    pivot = yhat[0]
    m_ = 0
    for i, y in enumerate(yhat):
        m = res[i][y]
        if m > m_:
            m_ = m
            pivot = yhat[i]
    return pivot, m_

def break_point(yhat):
    max_break = 0
    current_break = 0
    pivot = yhat[0]
    pivot_max = yhat[0]
    for v in yhat:
        if v == pivot:
            current_break += 1
            if current_break > max_break:
                max_break = current_break
                pivot_max = v
        else:
            current_break = 0
            pivot = v
    return pivot_max, max_break

def most_frequent(yhat, res):
    count_dict = {}
    for item in yhat:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1
    
    max_count = max(count_dict.values())
    most_frequent_items = [key for key, value in count_dict.items() if value == max_count]
    max_acc = find_max_acc(yhat, res)[0]
    p = break_point(yhat)[0]
    for i in most_frequent_items:
        if i == p:
            return i
        
    for i in most_frequent_items:
        if max_acc == i:
            return i

    return most_frequent_items[0]

def read_csv_to_list(file_path):
    """
    Đọc file CSV và chuyển đổi nó thành một danh sách.

    Parameters:
        file_path (str): Đường dẫn tới file CSV cần đọc.

    Returns:
        list: Danh sách các hàng trong file CSV. Mỗi hàng là một danh sách con chứa các giá trị tương ứng.
    """
    data_list = []
    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data_list.append(row)
    return data_list

# Sử dụng hàm để đọc file CSV và lưu vào biến
csv_file_path = '../Data_Round1/input.csv'
csv_data = read_csv_to_list(csv_file_path)

input_list = []
mp4_list = []
# In ra kết quả
for row in csv_data[1:]:
    mp4_list.append(row[0])
    a = row[0].split('/')[:-1]
    b = row[0].split('/')[-1].split(".")[0]
    a.append("pose")
    a.append(f"results_{b}.json")
    input_list.append('/'.join(a))

# Actions/exercises that we try to detect
actions = np.array(['chest fly machine', 'deadlift', 'hammer curl',
                    'incline bench press', 'pull Up', 'tricep dips', 
                    'decline bench press', 'leg raises', 'shoulder press',
                    'plank', 'leg extension', 'tricep Pushdown',
                    'bench press', 'lateral raise', 'squat',
                    'push-up', 'barbell biceps curl', 'russian twist',
                    'romanian deadlift', 'hip thrust', 'lat pulldown',
                    't bar row'
                    ])

# Videos are going to be this many frames in length
sequence_length = 60

data_poses = {}
min_d = 1000
for action in ["PIXELSPACE"]:
    data_poses[action] = {}
    for kk, filename in enumerate(input_list):
        if filename.endswith('.json'):
            file_path = filename
            key_file = mp4_list[kk]
            # Open the JSON file and load its contents
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            if len(data['instance_info']) >= sequence_length:
                # Do something with the data (for example, print it)
                data_poses[action][key_file] = []
                for frame in range(len(data['instance_info'])):
                    data_poses[action][key_file].append(data['instance_info'][frame]['instances'][0]['keypoints'])

label_map = {label:num for num, label in enumerate(actions)}

# Callbacks to be used during neural network training 
es_callback = EarlyStopping(monitor='val_loss', min_delta=5e-4, patience=10, verbose=0, mode='min')
lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=0, mode='min')
# # chkpt_callback = ModelCheckpoint(filepath=DATA_PATH, monitor='val_loss', verbose=0, save_best_only=True, 
#                                  save_weights_only=False, mode='min', save_freq=1)

# Optimizer
# opt = tf.keras.optimizers.Adam(learning_rate=0.01)
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)

# some hyperparamters
batch_size = 32
max_epochs = 500

# Set up Tensorboard logging and callbacks
NAME = f"ExerciseRecognition-AttnLSTM-{int(time.time())}"
log_dir = os.path.join(os.getcwd(), 'logs', NAME,'')
tb_callback = TensorBoard(log_dir=log_dir)

# callbacks = [tb_callback, es_callback, lr_callback, chkpt_callback]

def attention_block(inputs, time_steps):
    """
    Attention layer for deep neural network
    
    """
    # Attention weights
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    
    # Attention vector
    a_probs = Permute((2, 1), name='attention_vec')(a)
    
    # Luong's multiplicative score
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul') 
    
    return output_attention_mul

HIDDEN_UNITS = 256

# Input
inputs = Input(shape=(sequence_length, 51))

# Bi-LSTM
lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True))(inputs)

# Attention
attention_mul = attention_block(lstm_out, sequence_length)
attention_mul = Flatten()(attention_mul)

# Fully Connected Layer
x = Dense(2*HIDDEN_UNITS, activation='relu')(attention_mul)
x = Dropout(0.5)(x)

# Output
x = Dense(actions.shape[0], activation='softmax')(x)

# Bring it all together
AttnLSTM = Model(inputs=[inputs], outputs=x)

AttnLSTM.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# AttnLSTM.fit(X_train, y_train, batch_size=batch_size, epochs=max_epochs, validation_data=(X_val, y_val), callbacks=callbacks)

# Model map
models = {
    # 'LSTM': lstm, 
    'LSTM_Attention_128HUs': AttnLSTM, 
}

# Run model rebuild before doing this
for model_name, model in models.items():
    load_dir = os.path.join(os.getcwd(), f"{model_name}_f1.h5")
    model.load_weights(load_dir)

sequences, labels = [], []

new_sequences = {}
new_labels = {}
for action in ["PIXELSPACE"]:
    new_sequences[action] = {}
    new_labels[action] = {}
    for sequence in list(data_poses[action].keys()):
        new_sequences[action][sequence] = []
        new_labels[action][sequence] = []
        mod_d = len(data_poses[action][sequence])//sequence_length
        for m in range(mod_d):
            window = []
            for frame_num in range(m*sequence_length, (m+1)*sequence_length):         
                # LSTM input data
                res = np.asarray(data_poses[action][sequence][frame_num]).flatten()
                window.append(res)  
                
            sequences.append(window)
            new_sequences[action][sequence].append(window)

results_final = []
for action in ["PIXELSPACE"]:
    for video in list(new_sequences[action].keys()):
        X = np.array(new_sequences[action][video])

        for model_name, model in models.items():
            yhat_ = model.predict(X, verbose=0)
            yhat = np.argmax(yhat_, axis=1).tolist()
            results_final.append(actions[most_frequent(yhat, yhat_)])

output_csv = '../Data_Round1/output.csv'

# Mở tệp CSV để ghi
with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Ghi tiêu đề cho các cột
    writer.writerow(['video', 'Dự đoán'])
    
    # Ghi dữ liệu từ hai danh sách vào tệp CSV
    for item1, item2 in zip(mp4_list, results_final):
        writer.writerow([item1, item2])

print("Dữ liệu đã được ghi vào tệp: ", output_csv)