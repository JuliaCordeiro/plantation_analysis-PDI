from matplotlib import pyplot as plt
from turtle import color
from datetime import datetime
import cv2
import numpy as np
import os, sys

print(datetime.now())

def resize_image(image):
    scale_percent = 10
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimension = (width, height)
    image_resize = cv2.resize(image, dimension, interpolation = cv2.INTER_AREA)
    return image_resize

os.makedirs("./aux_image", exist_ok=True)
os.makedirs("./extract_leaf", exist_ok=True)
os.makedirs("./image_first_cut", exist_ok=True)
os.makedirs("./image_second_cut", exist_ok=True)

path = "./LuzBranca_Cima"
for filename in os.listdir(path):
    if filename.endswith('.jpg'):
        imagename = filename[:-4]

        img = cv2.imread(path + "/" + filename)
        # cv2.imshow('imagem', resize_image(img))

        # Canais HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # cv2.imshow('imagem hsv', resize_image(img_hsv))

        (img_H, img_S, img_V) = cv2.split(img_hsv)
        # cv2.imshow('imagem h', resize_image(img_H))
        # cv2.imshow('imagem s', resize_image(img_S))
        # cv2.imshow('imagem v', resize_image(img_V))

        #OPERAÇÃO
        (img_B, img_G, img_R) = cv2.split(img)
        # cv2.imshow('imagem b', resize_image(img_B))
        # cv2.imshow('imagem g', resize_image(img_G))
        # cv2.imshow('imagem r', resize_image(img_R))

        img_B = img_B - 0.8*img_V
        img_G = img_G - 0.8*img_V
        img_R = img_R - 0.8*img_V
        # cv2.imshow('imagem b', resize_image(img_B))
        # cv2.imshow('imagem g', resize_image(img_G))
        # cv2.imshow('imagem r', resize_image(img_R))

        new = cv2.merge([img_B, img_G, img_R])
        # cv2.imshow('imagem new', resize_image(new))
        cv2.imwrite(f"aux_image/{imagename}.jpg", new)

        new = cv2.imread(f"aux_image/{imagename}.jpg")
        # cv2.imshow('imagem new', resize_image(new))

        img_hsvnew = cv2.cvtColor(new, cv2.COLOR_BGR2HSV)
        # cv2.imshow('imagem hsv new', resize_image(img_hsvnew))

        (img_Hnew, img_Snew, img_Vnew) = cv2.split(img_hsvnew)
        # cv2.imshow('imagem h', resize_image(img_Hnew))
        # cv2.imshow('imagem s', resize_image(img_Snew))
        # cv2.imshow('imagem v', resize_image(img_Vnew))

        #SUAVIZAÇÃO
        soft = cv2.GaussianBlur(img_Hnew, (121, 121), 0) # aplica blur
        # cv2.imshow('imagem soft', resize_image(soft))

        #LIMIARIZAÇÃO
        # apply Otsu's automatic thresholding which automatically determines
        # the best threshold value
        (T, binI) = cv2.threshold(soft, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # print("[INFO] otsu's thresholding value: {}".format(T))

        #Aplicação de MORFOLOGIA
        kernel = np.ones((75, 75), np.uint8) 

        open = cv2.morphologyEx(binI, cv2.MORPH_OPEN, kernel,iterations=7)
        # cv2.imshow('imagem open', resize_image(open))

        # visualização da máscara aplicada à imagem original
        result_1 = cv2.bitwise_and(img_hsv, img_hsv, mask=open)
        result_2 = cv2.bitwise_and(img, img, mask=open)
        # cv2.imshow('imagem res1', resize_image(result_1))
        # cv2.imshow('imagem res2', resize_image(result_2))

        #ajuste de "bandas" apenas para usar o vstack para salvar o resultado
        binI = cv2.merge([binI, binI, binI])
        open = cv2.merge([open, open, open])
        # cv2.imshow('imagem binI', resize_image(binI))
        # cv2.imshow('imagem open', resize_image(open))

        #Juntando o resultado em um arquivo único
        imgResultado = np.vstack([
            np.hstack([binI, open]),
            np.hstack([result_1, result_2])
            ])
        # cv2.imshow('imagem result', resize_image(imgResultado))

        (resb,resg,resr) = cv2.split(result_2)
        # cv2.imshow('imagem b', resize_image(resb))
        # cv2.imshow('imagem g', resize_image(resg))
        # cv2.imshow('imagem r', resize_image(resr))

        #LIMIARIZAÇÃO
        # apply Otsu's automatic thresholding which automatically determines
        # the best threshold value
        (T, binIg) = cv2.threshold(resg, 100, 255, cv2.THRESH_BINARY)
        # print("[INFO] otsu's thresholding value: {}".format(T))

        binIgsoft = cv2.medianBlur(binIg, 11)
        # cv2.imshow('imagem binIg soft', resize_image(binIgsoft))

        cv2.imwrite(f"extract_leaf/{imagename}.jpg", result_2) 

''' End of extract leaf from image'''

pathMass = "./extract_leaf"
for filename in os.listdir(pathMass):
    if filename.endswith('.jpg'):
        imagename = filename[:-4]

        original = cv2.imread(path + "/" + filename)
        img = cv2.imread(pathMass + "/" + filename, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('imagem', resize_image(img))

        (T, binIm) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        # print("[INFO] otsu's thresholding value: {}".format(T))
        # print(f"Thresh: {binIm}")
        
        height, width = binIm.shape[:2]
        # print(f"H: {height}, W: {width}")

        mass = 0
        x_size = 0.0
        y_size = 0.0

        for i in range(width):
            for j in range(height):
                if not binIm[j][i]:
                    mass += 1
                    x_size += i
                    y_size += j
        # print(f"mass: {mass}, X: {x_size}, Y: {y_size}")

        x_size = x_size/mass
        y_size = y_size/mass
        # print(f"X: {x_size}, Y: {y_size}")

        x = int(x_size)
        y = int(y_size)
        # print(f"X: {x}, Y: {y}")

        cut = original[(y-100):(y+100), (x-100):(x+100)]
        # cv2.imshow("Recorte da imagem", cut)
        cv2.imwrite(f"image_first_cut/{imagename}.jpg", cut) 

        width_orig = original.shape[1]
        height_orig = original.shape[0]

        y_center = int(height_orig // 2)
        x_center = int(width_orig // 2)

        cut_2 = original[(y_center-100):(y_center+100), (x_center-100):(x_center+100)]
        # cv2.imshow("Recorte 2 da imagem", cut_2)
        cv2.imwrite(f"image_second_cut/{imagename}.jpg", cut_2)

''' End of cut image mass center'''

print(datetime.now())
#espera para fechar programa
cv2.waitKey(0) #espera pressionar qualquer tecla