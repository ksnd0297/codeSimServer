import os
import pprint

from PIL import Image
from extract_keywords import extract_cpp_keywords
from myTurtle import Turtle

# cpp_keywords = ['while', 'break', 'else' ,'if', 'for', 'int', 'double', 'char', 'new', 'void']

# /code 폴더의 cpp 코드들 다 가져와서 turtle 로 그림 그리는 코드 
# 여기서 svg랑 png 한 세트씩 만들어짐

def drawCode(codeList):
    for index in range(0, len(codeList)):
        code_keywords = extract_cpp_keywords(codeList[index])

        T = Turtle()
        T.setKeywordsPosition(False)

        for i in range(0, len(code_keywords) - 1):
            T.drawLine(code_keywords[i], code_keywords[i+1])

        T.done('code' + str(index))

# 이미지 자르는 함수
def crop_image(image_path, file_name, keep_ratio):
    # 이미지 로드
    file_path = image_path + '/' + file_name
    image = Image.open(file_path)

    # 이미지의 너비와 높이 가져오기
    width, height = image.size
    
    # 자르기를 원하는 비율 계산
    crop_ratio = 1 - keep_ratio
    
    # 자를 영역 계산
    left = int(width * crop_ratio / 2)
    upper = int(height * crop_ratio / 2)
    right = int(width * (1 - crop_ratio / 2))
    lower = int(height * (1 - crop_ratio / 2))
    
    # 이미지 자르기
    cropped_image = image.crop((left, upper, right, lower))

    # 자른 이미지 저장
    cropped_image.save("./crop_image/" + file_name)


def cleanUp():
    imageList = os.listdir("Image")

    print(imageList)

    # 아래 for 문은 /image 폴더에서 추후 사용하지 않을 svg 삭제하는 부분
    for file in imageList:
        if len(file.split('.')) == 2:
            # print(i, file)
            # i+=1
            filename = file.split('.')[1]
            if(filename == "svg"):
                os.remove("image/{}".format(file))


    imagePath = os.getcwd() + '/Image'
    imageList = os.listdir(imagePath)

    # 아래 for 문은 crop_image 함수 호출하여 이미지 사이즈 리사이징 하는 부분
    for fileName in imageList:
        if len(fileName.split('.')) == 2:
            # image파일 경로, image파일 이름, 이미지 사이즈 퍼센트
            # 이미지퍼센트는 60%를 남긴다는 뜻
            crop_image(imagePath, fileName, 0.6) 
        

