#Test Test Test
import os
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from scipy.spatial import distance

# 모델 URL
#ver1
model_url_v1_0 = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
model_url_v1_1 = "https://tfhub.dev/google/efficientnet/b1/feature-vector/1"
model_url_v1_2 = "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1"
model_url_v1_3 = "https://tfhub.dev/google/efficientnet/b3/feature-vector/1"
model_url_v1_4 = "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1"
model_url_v1_5 = "https://tfhub.dev/google/efficientnet/b5/feature-vector/1"
model_url_v1_6 = "https://tfhub.dev/google/efficientnet/b6/feature-vector/1"
model_url_v1_7 = "https://tfhub.dev/google/efficientnet/b7/feature-vector/1"

#ver2
model_url_v2_0 = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2"
model_url_v2_1 = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2"
model_url_v2_2 = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/feature_vector/2"
model_url_v2_3 = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2"

# 하이퍼파라미터
size = 6
fomat = 'png'
#IMAGE_SHAPE = (224, 224)
IMAGE_SHAPE = (1024, 1024)
layer = hub.KerasLayer(model_url_v2_0) #모델은 V1이 0~7까지, V2는 0~3까지 존재
model = tf.keras.Sequential([layer])
metric1 = 'euclidean'

# 사이즈 변환 함수
def replicate_image(image_path, size, fomat, ii):
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
    new_image.save(f'resizing_test{ii}_{size}X{size}.{fomat}')

# 이미지 추출 함수
def extract(file):
    file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
    file = np.stack((file,)*3, axis=-1)
    file = np.array(file)/255.0
    embedding = model.predict(file[np.newaxis, ...])
    vgg16_feature_np = np.array(embedding)
    flattended_feature = vgg16_feature_np.flatten()
    print(flattended_feature)
    return flattended_feature

# 거리 계산 함수
def calculate_distances(test1, test2, test3, metric1):
    print('euclidean')
    dc = distance.cdist([test1], [test1], metric1)[0]
    print(100-dc)
    dc1 = distance.cdist([test1], [test2], metric1)[0]
    print(100-dc1, end='')
    dc2 = distance.cdist([test1], [test3], metric1)[0]
    print(100-dc2, end='')
    dc3 = distance.cdist([test2], [test3], metric1)[0]
    print(100-dc3, end='')

def execute():
    # 이미지 복제 및 size X size 형식의 이미지 생성
    for ii in range(1, 4):
        image_path = f'test{ii}.{fomat}'
        replicate_image(image_path, size, fomat, ii)

    # 변환된 이미지 추출
    test1 = extract(f'resizing_test1_{size}X{size}.png')
    test2 = extract(f'resizing_test2_{size}X{size}.png')
    test3 = extract(f'resizing_test3_{size}X{size}.png')

    # 결과 
    calculate_distances(test1, test2, test3, metric1)    