#Test Test Test
import os
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from scipy.spatial import distance

#ver2
model_url_v2_0 = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2"
model_url_v2_1 = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2"
model_url_v2_2 = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/feature_vector/2"
model_url_v2_3 = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2"

# # 하이퍼파라미터
size = 6
fomat = 'png'
# #IMAGE_SHAPE = (224, 224)
IMAGE_SHAPE = (1024, 1024)
layer = hub.KerasLayer(model_url_v2_3) #모델은 V1이 0~7까지, V2는 0~3까지 존재
print('layer', layer)
model = tf.keras.Sequential([layer])
metric1 = 'euclidean'

# 사이즈 변환 함수
def replicate_image(image_path, size, file):
    image = Image.open(image_path)
    width, height = image.size
    new_width = width * size
    new_height = height * size
    new_image = Image.new('RGB', (new_width, new_height))

    for i in range(size):
        for j in range(size):
            paste_x = i * width
            paste_y = j * height
            new_image.paste(image, (paste_x, paste_y))

    new_image.save(f'ReImage/{file}')

# 이미지 추출 함수
def extract(file):
    file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
    file = np.stack((file,)*3, axis=-1)
    file = np.array(file)/255.0
    embedding = model.predict(file[np.newaxis, ...])
    vgg16_feature_np = np.array(embedding)
    flattended_feature = vgg16_feature_np.flatten()
    return flattended_feature

# 거리 계산 함수
def calculate_distances(code, metric1):
    arr = []

    for i in range(0, len(code)):
        for j in range(i +1, len(code)):
            dc = distance.cdist([code[i]], [code[j]], metric1)[0]

            arr.append({
                'from' : i,
                'to' : j,
                'result' : 100 - dc[0]
            })


    return arr

def execute():
    # 이미지 복제 및 size X size 형식의 이미지 생성
    path = "./crop_Image"
    fileList = os.listdir(path)

    for file in fileList:
        image_path = path + '/' + file
        
        replicate_image(image_path, size, file)

    # 변환된 이미지 추출
    code = []

    path ="./ReImage"
    fileList = os.listdir(path)
    for file in fileList:
        code.append(extract(path + '/' + file))

    # 결과 
    arr = calculate_distances(code, metric1)

    return arr