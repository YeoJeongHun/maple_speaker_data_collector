import io # 파일을 읽고 쓰기위한 모듈
import os # os의 기능을 사용하기 위한 모듈

# Imports the Google Cloud client library
from google.cloud import vision
# from google.cloud.vision import types

client = vision.ImageAnnotatorClient()
client.annotate_image('https://foo.com/image.jpg')
