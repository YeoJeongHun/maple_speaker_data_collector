from PIL import ImageGrab
import cv2
# import keyboard
# import mouse
import numpy as np
import sys
import winsound as sd

temp_img = cv2.imread("speaker.jpg")
ch_img = cv2.imread('ch.jpg')
temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)

rungbbu = cv2.imread('rungbbu.jpg')
rungbbu1 = cv2.imread('rungbbu1.jpg')

h, w = temp_img.shape
ch_h, ch_w = ch_img.shape[0:2]
rungbbu_h, rungbbu_w = rungbbu.shape[0:2]
ch_pt = (0, 0)
global content

def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 1000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

while True:
    image = cv2.cvtColor(np.array(ImageGrab.grab(bbox = (0, 0, 2560, 1440), all_screens=True)),cv2.COLOR_BGR2RGB)

    if temp_img is None or image is None:
        print('Image load failed!')
        sys.exit()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_h, image_w = image.shape[0:2]

    result = cv2.matchTemplate(image, temp_img, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(result)
    x, y = maxloc

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    if maxv > 0.9 :
        image = cv2.rectangle(image, (x, y), (x+400, y+170), (0,0,255), 2)
        main_chat = cv2.cvtColor(np.array(ImageGrab.grab(bbox = (x, y+122, x+380, y+180), all_screens=True)),cv2.COLOR_BGR2RGB)
        ch_res = cv2.matchTemplate(main_chat,ch_img, cv2.TM_CCOEFF_NORMED)
        loc = np.where(ch_res > 0.8)
        for ch_pt in zip(*loc[::-1]):
            print(ch_pt)
            # main_chat = cv2.rectangle(main_chat, (ch_pt[0], ch_pt[1]), (ch_pt[0]+ch_w, ch_pt[1]+ch_h), (0,255,0), 1)
        ch_num = main_chat[ch_pt[1]-2:ch_pt[1]+12, ch_pt[0]-3:ch_pt[0]+35].copy()
        titme_nickname = main_chat[ch_pt[1]-3:ch_pt[1]+14, 0:ch_pt[0]-5].copy()
        content = main_chat[ch_pt[1]-3:ch_pt[1]+14, ch_pt[0]+46:380].copy()
        # if ch_pt[1] < 25 :
        #     content = np.concatenate([content,main_chat[ch_pt[1]+13:ch_pt[1]+30, 20:]],axis=1)
        # if ch_pt[1] < 10 :
        #     content = np.concatenate([content,main_chat[ch_pt[1]+31:ch_pt[1]+48, 20:]],axis=1)
        # content = np.concatenate([res_last_chat,res_last_chat1],axis=1)
    rungbbu_result = cv2.matchTemplate(content, rungbbu, cv2.TM_CCOEFF_NORMED)
    _, rungbbu_maxv, _, rungbbu_maxloc = cv2.minMaxLoc(rungbbu_result)
    rungbbu1_result = cv2.matchTemplate(content, rungbbu1, cv2.TM_CCOEFF_NORMED)
    _, rungbbu1_maxv, _, rungbbu1_maxloc = cv2.minMaxLoc(rungbbu1_result)

    # if rungbbu_maxv > 0.9 :
    #     content = cv2.rectangle(content, (rungbbu_maxloc[0], rungbbu_maxloc[1]), (rungbbu_maxloc[0]+rungbbu_w, rungbbu_maxloc[1]+rungbbu_h), (255,255,0), 1)
    #     beepsound()
    # if rungbbu1_maxv > 0.9 :
    #     content = cv2.rectangle(content, (rungbbu1_maxloc[0], rungbbu1_maxloc[1]), (rungbbu1_maxloc[0]+rungbbu_w, rungbbu1_maxloc[1]+rungbbu_h), (255,255,0), 1)
    #     beepsound()

    image = cv2.resize(image, (int(image_w/4), int(image_h/4)))

    if titme_nickname is None or main_chat is None or image is None :
        print('Image load failed!')
        continue

    cv2.imshow('image', image)
    cv2.imshow('main_chat', main_chat)
    # cv2.imshow('titme_nickname',titme_nickname)
    cv2.imshow('content',content)
    cv2.imshow('ch_num',ch_num)

    if cv2.waitKey(50) == 27 :
        break


cv2.destroyAllWindows()

