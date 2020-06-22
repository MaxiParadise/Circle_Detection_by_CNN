'''
  Tensorflow 2.2 + Python3.7環境にて動作確認
'''
import os
import sys
import numpy as np
import csv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16
from PIL import Image, ImageDraw

# --- Parameter Setting ---

# 最大エポック数
MAX_EPOCH = 100

# 入力データサイズ
IMAGE_W = 224
IMAGE_H = 224

# 出力パラメータ数
OUTPUT_PARAMS = 3 # x, y, radius

# データフォルダ
TEST_DIR = 'dataset/image/test'
TEST_ANNOT_DATA = 'dataset/annotation/test.csv'

# テスト結果保存
RESULT_DIR = 'test_result'
os.makedirs(RESULT_DIR, exist_ok=True)

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


def Write_Result(img_dir_path, annot_csv_path, result):
    line_len = 7
    with open(annot_csv_path) as f:
        for csv_line, result in zip(csv.reader(f), result_list):
            fname, dummy_x, dummy_y, dummy_r = csv_line
            fpath = os.path.join(img_dir_path, fname)
            img = Image.open(fpath)
            x, y, radius = result * IMAGE_H
            draw = ImageDraw.Draw(img)
            draw.line((x-line_len, y-line_len, x+line_len, y+line_len), fill=(255,0,0), width=3)
            draw.line((x+line_len, y-line_len, x-line_len, y+line_len), fill=(255,0,0), width=3)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=(255,0,0), width=3)
            img.save(os.path.join(RESULT_DIR,fname))



# --- Main ---
if __name__ == '__main__':
    argl = len(sys.argv)
    if argl != 2:
        print('python this.py <model_path>')
        sys.exit()

    # 全体Model作成
    model_path = sys.argv[1]
    model = load_model(model_path)

    # Summary表示
    model.summary()


    # --- Train Setting ---

    # 学習セットのロード
    X_test, Y_test = Read_Dataset(TEST_DIR, TEST_ANNOT_DATA)

    # 正規化(colorデータなので0-255)
    X_test = np.array(X_test, dtype=np.float32) / 255.0

    result_list = model.predict(X_test)

    for i,(result, y_dat) in enumerate(zip(result_list, Y_test)):
        x, y, radius = result * IMAGE_H
        y_x, y_y, y_radius = y_dat
        print('test{:2d} predict x:{:6.2f}, y:{:6.2f}, r:{:6.2f} Correct x:{:6.2f}, y:{:6.2f}, r:{:6.2f}'.format(i,x,y,radius,y_x,y_y,y_radius))

    r_y_diff = np.abs(Y_test-result_list*IMAGE_H)
    r_y_err = r_y_diff.sum(axis=0) / len(Y_test)
    print('Average Error x:{:.2f}, y:{:.2f}, r:{:.2f}'.format(r_y_err[0],r_y_err[1],r_y_err[2]))

    Write_Result(TEST_DIR, TEST_ANNOT_DATA, result_list)
