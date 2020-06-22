'''
  Tensorflow 2.2 + Python3.7環境にて動作確認
'''
import os
import numpy as np
import csv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from PIL import Image

# --- Parameter Setting ---

# 最大エポック数
MAX_EPOCH = 100

# 打ち切り判断数
ES_PATIENCE = 15
# SaveBestOnly
SAVE_BEST_ONLY = True

# バッチサイズ
BATCH_SIZE = 16

# 入力データサイズ
IMAGE_W = 224
IMAGE_H = 224

# 出力パラメータ数
OUTPUT_PARAMS = 3 # x, y, radius

# 初期学習率
LEARN_RATE = 0.0001

# データフォルダ
TRAIN_DIR = 'dataset/image/train'
VALID_DIR = 'dataset/image/valid'
TRAIN_ANNOT_DATA = 'dataset/annotation/train.csv'
VALID_ANNOT_DATA = 'dataset/annotation/valid.csv'

# チェックポイント保存フォルダ
CP_DIR = 'checkpoint'
DEBUG_DIR = 'debug'
os.makedirs(CP_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

def Read_Dataset(img_dir_path, annot_csv_path):
    x_list = []
    y_list = []
    
    with open(annot_csv_path) as f:
        for fname, x, y, radius in csv.reader(f):
            fpath = os.path.join(img_dir_path, fname)
            img = Image.open(fpath).resize((IMAGE_H, IMAGE_W), Image.LINEAR)
            x_list.append(np.reshape(np.array(img, 'f'), (IMAGE_H, IMAGE_W, 3)))
            y_list.append([float(x), float(y), float(radius)])

    return x_list, y_list

# --- Main ---
if __name__ == '__main__':

    # --- Model Setting ---
    # Input layer 作成
    input_tensor = Input(shape=(IMAGE_H, IMAGE_W, 3))

    # Base Model 作成
    #base_model = MobileNetV2(include_top=False, input_tensor=input_tensor)
    base_model = VGG16(include_top=False, input_tensor=input_tensor)

    # Output layer 作成
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    # 
    predictions = Dense(OUTPUT_PARAMS, activation='linear')(x)

    # 全体Model作成
    model = Model(inputs=base_model.input, outputs=predictions)

    # Optimizer 選択
    model.compile(optimizer=Adam(lr=LEARN_RATE), loss='mse', metrics=['accuracy'])

    # Summary表示
    model.summary()


    # --- Train Setting ---

    # 学習セットのロード
    X_train, Y_train = Read_Dataset(TRAIN_DIR, TRAIN_ANNOT_DATA)
    X_valid, Y_valid = Read_Dataset(VALID_DIR, VALID_ANNOT_DATA)


    # 正規化(colorデータなので0-255)
    X_train = np.array(X_train, dtype=np.float32) / 255.0
    X_valid = np.array(X_valid, dtype=np.float32) / 255.0

    # 正規化(x,y,radiusは画像サイズで正規化)
    Y_train = np.array(Y_train, dtype=np.float32)/IMAGE_H
    Y_valid = np.array(Y_valid, dtype=np.float32)/IMAGE_H

    # Callback選択
    cb_funcs = []

    # Checkpoint作成設定
    check_point = ModelCheckpoint(filepath = os.path.join(CP_DIR, 'epoch{epoch:03d}-{val_loss:.5f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=SAVE_BEST_ONLY, mode='auto')
    cb_funcs.append(check_point)

    # Early-stopping Callback設定
    if ES_PATIENCE >= 0:
        early_stopping = EarlyStopping(patience=ES_PATIENCE, verbose=1)
        cb_funcs.append(early_stopping)

    # モデル訓練実行
    history = model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCH,
        verbose=1,
        validation_data=(X_valid, Y_valid),
        callbacks=cb_funcs
    )
