'''
  Tensorflow 2.2 + Python3.7環境にて動作確認
'''
import os
import random
from PIL import Image, ImageDraw
from tqdm import tqdm


#defines
OUT_IMG_DIR = 'dataset/image'
OUT_ANNOT_DIR = 'dataset/annotation'
TRAIN_STR = 'train'
VALID_STR = 'valid'
TEST_STR = 'test'

DATA_TRAIN_NUM = 1000
DATA_VALID_NUM = 200
DATA_TEST_NUM = 100

IMG_SIZE = 224
RAD_MAX = 50
RAD_MIN = 20


# Generate Random Circle Data
def generate_circle_data(num, target_name):
    # make annotation file
    with open(os.path.join(OUT_ANNOT_DIR,'{}.csv'.format(target_name)), mode='w') as f:
        #generate image data
        for i in tqdm(range(num)):
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
            # random radius
            radius = random.randint(RAD_MIN, RAD_MAX)
            # random color
            rr = random.randint(0, 128)
            gg = random.randint(0, 128)
            bb = random.randint(0, 128)
            # random position
            x = random.randint(radius,IMG_SIZE-radius)
            y = random.randint(radius,IMG_SIZE-radius)

            # draw to canvas
            draw = ImageDraw.Draw(img)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(rr,gg,bb))
            # save the image
            img_fname ='{}_{:06d}.png'.format(target_name,i)
            img.save(os.path.join(OUT_IMG_DIR,target_name,img_fname))

            # write to annotation file
            f.write('{},{:d},{:d},{:d}\n'.format(img_fname,x,y,radius))


# ----- main -----

# Make output dir
os.makedirs(os.path.join(OUT_IMG_DIR,TRAIN_STR), exist_ok=True)
os.makedirs(os.path.join(OUT_IMG_DIR,VALID_STR), exist_ok=True)
os.makedirs(os.path.join(OUT_IMG_DIR,TEST_STR), exist_ok=True)
os.makedirs(os.path.join(OUT_ANNOT_DIR), exist_ok=True)

# generate as fixed random data
random.seed(12345)

# Call function
generate_circle_data(DATA_TRAIN_NUM, TRAIN_STR)
generate_circle_data(DATA_VALID_NUM, VALID_STR)
generate_circle_data(DATA_TEST_NUM, TEST_STR)
