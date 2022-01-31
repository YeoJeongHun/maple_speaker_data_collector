from PIL import ImageGrab
import cv2
import numpy as np
import sys
import winsound as sd
import io 
import os 
import threading
import datetime
from socket import *
# Imports the Google Cloud client library 
from google.cloud import vision 

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision_api_key.json'

# Instantiates a client 
client = vision.ImageAnnotatorClient()

HOST = '127.0.0.1'
PORT = 8025
BUFSIZE = 1024

clientSocket = socket(AF_INET, SOCK_STREAM)

temp_img = cv2.imread("speaker.jpg")
ch_img = cv2.imread('ch.jpg')
temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
last_chet_before = None
last_chet_before_flag = True

h, w = temp_img.shape
ch_h, ch_w = ch_img.shape[0:2]
ch_pt = (0, 0)

def processVisionApi(content):
    textArr = []
    
    image = vision.Image(content=content) 

    # Performs label detection on the image file 
    response = client.text_detection(image=image) 
    texts = response.text_annotations 

    print('text:') 
    for text in texts: 
        # print(text.description)
        textArr.append((text.description).replace('\n',''))
    originText = "".join(textArr[1:])
    # print('전체 : ', originText)

    if originText.rfind('CH') > -1 :
        text_ch = originText[originText.rfind('CH'):originText.rfind('CH')+6]
        # print('체널 : ', text_ch)
    else :
        text_ch = '없음'

    if -1 < originText.find('>') < 15 :
        text_simbol = originText[:originText.rfind('>')+1]
        # print('칭호 : ', text_simbol)
    else :
        text_simbol = '없음'
    
    if -1 < originText.find('[') < 25 or -1 < originText.find(':') < 25:
        if -1 < originText.find('>') < 15 :
            if -1 < originText.find(':') < 25 :
                text_nickname = originText[originText.find('>')+1:originText.find(':')]
                if -1 < originText.find('[') < 25 :
                    text_content = originText[originText.find(']')+1:]
                    text_link = originText[originText.find('[')+1:originText.find(']')]
                else :
                    text_content = originText[originText.find(':')+1:]
                    text_link = '없음'
            elif -1 < originText.find('[') < 25 :
                text_nickname = originText[originText.find('>')+1:originText.find('[')]
                text_content = originText[originText.find(']')+1:]
                text_link = originText[originText.find('[')+1:originText.find(']')]
            else :
                text_nickname = '없음'
                text_content = originText
                text_link = '없음'
        else :
            if -1 < originText.find(':') < 25 :
                text_nickname = originText[:originText.find(':')]
                if -1 < originText.find('[') < 25 :
                    text_content = originText[originText.find(']')+1:]
                    text_link = originText[originText.find('[')+1:originText.find(']')]
                else :
                    text_content = originText[originText.find(':')+1:]
                    text_link = '없음'
            elif -1 < originText.find('[') < 25 :
                text_nickname = originText[:originText.find('[')]
                text_content = originText[originText.find(']')+1:]
                text_link = originText[originText.find('[')+1:originText.find(']')]
            else :
                text_nickname = '없음'
                text_content = originText
                text_link = '없음'
        # print('닉네임 : ', text_nickname)
        # print('내용 : ', text_content)
        # print('링크 : ', text_link)
    else :
        text_nickname = '없음'
        text_link = '없음'
        text_content = originText
    
    now = datetime.datetime.now()
    now = now.strftime('%Y-%m-%d %H:%M:%S')
    text_json = {
        "text_origin" : originText,
        "text_simbol" : text_simbol,
        "text_nickname" : text_nickname,
        "text_ch" : text_ch,
        "text_link" : text_link,
        "text_content" : text_content,
        "text_time" : now,
    }
    print(text_json)

    try:
        clientSocket.connect((HOST, PORT))
        clientSocket.sendall(bytes(text_json, 'UTF-8'))
    except Exception as e :
        print(e)
        
    


while True:
    image = cv2.cvtColor(np.array(ImageGrab.grab(bbox = (0, 0, 2560, 1440), all_screens=True)),cv2.COLOR_BGR2RGB)

    if temp_img is None or image is None or ch_img is None:
        print('Image load failed!')
        sys.exit()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_h, image_w = image.shape[0:2]

    result = cv2.matchTemplate(image, temp_img, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(result)
    x, y = maxloc

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    if maxv > 0.9 :
        main_chat = cv2.cvtColor(np.array(ImageGrab.grab(bbox = (x, y+122, x+380, y+172), all_screens=True)),cv2.COLOR_BGR2RGB)
        ch_res = cv2.matchTemplate(main_chat,ch_img, cv2.TM_CCOEFF_NORMED)
        loc = np.where(ch_res > 0.5)
        for ch_pt in zip(*loc[::-1]):
            ch_pt = ch_pt
            # print(ch_pt)
            # main_chat = cv2.rectangle(main_chat, (ch_pt[0], ch_pt[1]), (ch_pt[0]+ch_w, ch_pt[1]+ch_h), (0,255,0), 1)
        last_chet_view = main_chat[ch_pt[1]-2:, :].copy()
        last_chet_h, last_chet_w = last_chet_view.shape[0:2]
        if last_chet_h > 40 :
            last_chet = np.concatenate([main_chat[ch_pt[1]-2:ch_pt[1]+13, :],main_chat[ch_pt[1]+14:ch_pt[1]+29, 27:],main_chat[ch_pt[1]+30:ch_pt[1]+45, 27:]],axis=1)
        elif last_chet_h > 25 :
            last_chet = np.concatenate([main_chat[ch_pt[1]-2:ch_pt[1]+13, :],main_chat[ch_pt[1]+14:ch_pt[1]+29, 27:]],axis=1)
        else :
            last_chet = main_chat[ch_pt[1]-2:, :].copy()
        # last_chet = cv2.rectangle(last_chet, (ch_pt[0], ch_pt[1]), (ch_pt[0]+ch_w, ch_pt[1]+ch_h), (0,255,0), 1)

    if last_chet_before_flag :
        last_chet_before = last_chet_view.copy()
        last_chet_before_flag = False

    result = cv2.matchTemplate(last_chet_view, last_chet_before, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(result)

    if maxv > 0.9 :
        print('같음', maxv)
    else :
        print('바뀜')  
        
        cv2.imwrite('./last_chet_img.jpg', last_chet)

        file_name = os.path.abspath('last_chet_img.jpg') 

        # Loads the image into memory 
        with io.open(file_name, 'rb') as image_file: 
            content = image_file.read()
        
        try:
            thr = threading.Thread(target=processVisionApi, args=(content,))
            thr.start()
        except:
            print('error')

    last_chet_before = last_chet_view.copy()


    # image = cv2.resize(image, (int(image_w/4), int(image_h/4)))
    # cv2.imshow('image', image)
    cv2.imshow('main_chat', main_chat)
    cv2.imshow('last_chet', last_chet)

    if cv2.waitKey(10) == 27 :
        break


cv2.destroyAllWindows()

