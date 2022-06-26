from matplotlib import pyplot as plt
from turtle import color
import cv2
import numpy as np
import os, sys

#Função de Convolução 2D
def convolution2d(imagem, kernel):
    m, n = kernel.shape
    c = m % 2 
    #garantir que o kernel é impar e do quadrado
    if(m == n) & (c != 0):
        y, x = imagem.shape
        nova_imagem = np.zeros((y,x))
        #ignorar os pontos de borda que extrapolariam
        y = y - m + 1
        x = x - m + 1
        for i in range(y):
            for j in range(x):
                nova_imagem[i+c][j+c] = np.sum(imagem[i:i+m, j:j+m]*kernel)
    return nova_imagem

def resize_image(image):
    scale_percent = 10
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimension = (width, height)
    image_resize = cv2.resize(image, dimension, interpolation = cv2.INTER_AREA)
    return image_resize


path = "./test"
for filename in os.listdir(path):
    if filename.endswith('.jpg'):
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

        nova = cv2.merge([img_B, img_G, img_R])
        # cv2.imshow('imagem nova', resize_image(nova))
        cv2.imwrite(f"imagens_auxiliares/${filename}.jpg", nova)

        nova = cv2.imread(f"imagens_auxiliares/${filename}.jpg")
        # cv2.imshow('imagem nova', resize_image(nova))

        img_hsvnova = cv2.cvtColor(nova, cv2.COLOR_BGR2HSV)
        # cv2.imshow('imagem hsv nova', resize_image(img_hsvnova))

        (img_Hnova, img_Snova, img_Vnova) = cv2.split(img_hsvnova)
        # cv2.imshow('imagem h', resize_image(img_Hnova))
        # cv2.imshow('imagem s', resize_image(img_Snova))
        # cv2.imshow('imagem v', resize_image(img_Vnova))

        #SUAVIZAÇÃO
        suave = cv2.GaussianBlur(img_Hnova, (121, 121), 0) # aplica blur
        # cv2.imshow('imagem suave', resize_image(suave))

        #LIMIARIZAÇÃO
        # apply Otsu's automatic thresholding which automatically determines
        # the best threshold value
        (T, binI) = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        print("[INFO] otsu's thresholding value: {}".format(T))

        #Aplicação de MORFOLOGIA
        kernel = np.ones((75, 75), np.uint8) 

        abertura = cv2.morphologyEx(binI, cv2.MORPH_OPEN, kernel,iterations=7)
        # cv2.imshow('imagem abertura', resize_image(abertura))

        # visualização da máscara aplicada à imagem original
        resultado1 = cv2.bitwise_and(img_hsv, img_hsv, mask=abertura)
        resultado2 = cv2.bitwise_and(img, img, mask=abertura)
        # cv2.imshow('imagem res1', resize_image(resultado1))
        # cv2.imshow('imagem res2', resize_image(resultado2))

        #ajuste de "bandas" apenas para usar o vstack para salvar o resultado
        binI = cv2.merge([binI, binI, binI])
        abertura = cv2.merge([abertura, abertura, abertura])
        # cv2.imshow('imagem binI', resize_image(binI))
        # cv2.imshow('imagem abertura', resize_image(abertura))

        #Juntando o resultado em um arquivo único
        imgResultado = np.vstack([
            np.hstack([binI, abertura]),
            np.hstack([resultado1, resultado2])
            ])
        # cv2.imshow('imagem result', resize_image(imgResultado))

        (resb,resg,resr) = cv2.split(resultado2)
        # cv2.imshow('imagem b', resize_image(resb))
        # cv2.imshow('imagem g', resize_image(resg))
        # cv2.imshow('imagem r', resize_image(resr))

        #LIMIARIZAÇÃO
        # apply Otsu's automatic thresholding which automatically determines
        # the best threshold value
        (T, binIg) = cv2.threshold(resg, 100, 255, cv2.THRESH_BINARY)
        print("[INFO] otsu's thresholding value: {}".format(T))

        binIgsuave = cv2.medianBlur(binIg, 11)
        # cv2.imshow('imagem binIg Suave', resize_image(binIgsuave))

''' End of extract leaf from image'''


        # cv2.imwrite(f"folha_extraida/${filename}.jpg", ) 
        

#espera para fechar programa
cv2.waitKey(0) #espera pressionar qualquer tecla