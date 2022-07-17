import cv2
import numpy as np
# import os
# import math
# import imutils


def histograms(thresh1, original):
    threshM2 = thresh1.copy()

    contornos, hierarchy = cv2.findContours(threshM2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    jan_contorno = original.copy()
    cor_contorno = (0, 255, 0)  # verde

    max_area = []

    for i in range(len(contornos)):
        max_area.append(cv2.contourArea(contornos[i]))
    print('max_area: ', max_area, max(max_area))

    for i in range(len(contornos)):
        if max_area[i] != max(max_area):
            x, y, w, h = cv2.boundingRect(contornos[i])
            cv2.rectangle(threshM2, (x, y), (x + w, y + h), 0, -1)

    image = threshM2
    height, width = image.shape[:2]

    cv2.imshow("M2", threshM2)


    histH = []
    histH_der = []
    for c in range(width):
        conta = 0
        for l in range(height):
            conta += image[l, c]
        histH.append(conta / height)
        histH_der.append(histH[c] - histH[
            c - 1])  # penso que tenho um epqueno bug por causa do -1 inicial mas que neste caso nao faz diferença

    histV = []
    histV_der = []
    for l in range(height):
        conta = 0
        for c in range(width):
            conta += image[l, c]
        histV.append(conta / width)
        histV_der.append(histV[l] - histV[
            l - 1])  # penso que tenho um epqueno bug por causa do -1 inicial mas que neste caso nao faz diferença

    HH = np.zeros((101, width, 3), np.uint8)
    for c in range(width):
        cv2.line(HH, (c, 100 - int(histH[c] * 100 / 255)), (c, 100), (0, 0, 255))  # (direita, baixo)
    
    HV = np.zeros((height, 101, 3), np.uint8)  # altura, largura, canais
    for l in range(height):
        cv2.line(HV, (int(histV[l] * 100 / 255), l), (0, l), (0, 255, 0))  # (direita, baixo)
    
    HH_der = np.zeros((101, width, 3), np.uint8)
    for c in range(width):
        cv2.line(HH_der, (c, 50 - int(histH_der[c] * 100 / 255)), (c, 50), (0, 0, 255))  # (direita, baixo)

    HV_der = np.zeros((height, 101, 3), np.uint8)  # altura, largura, canais
    for l in range(height):
        cv2.line(HV_der, (50 - int(histV_der[l] * 100 / 255), l), (50, l), (0, 255, 0))  # (direita, baixo)
        
    cv2.imshow('HH', HH)
    cv2.imshow('HV', HV)
    cv2.imshow('HH_der', HH_der)
    cv2.imshow('HV_der', HV_der)

    histH_alt = []
    histV_alt = []
    histH_der_alt = []
    histV_der_alt = []

    for c in range(width):
        histH_der_alt.append(int(histH_der[c] * 100 / 255))
        histH_alt.append(int(histH[c] * 100 / 255))
    for l in range(height):
        histV_der_alt.append(int(histV_der[l] * 100 / 255))
        histV_alt.append(int(histV[l] * 100 / 255))

    for c in range(width):
        print("%.1f" % histH_der[c], "" ", ", end="")
    print()
    for c in range(width):
        print(float(int(histH_der[c] * 100 / 255)), ", ", end="")
    print()

    # linhas derivadas cinzento escuro
    for c in range(width):
        if abs(histH_der_alt[c]) > 1 and abs(histH_der_alt[c - 1]) <= 1:
            print("lineH_der inicial", c)
            cv2.line(threshM2, (c, 0), (c, threshM2.shape[0]), 80)  # (direita, baixo)
        elif abs(histH_der_alt[c]) > 1 and abs(histH_der_alt[c + 1]) <= 1:
            print("lineH_der final", c)
            cv2.line(threshM2, (c, 0), (c, threshM2.shape[0]), 80)  # (direita, baixo)

    for l in range(height):
        if abs(histV_der_alt[l]) > 0 and abs(histV_der_alt[l - 1]) <= 0:
            print("lineV_der inicial", l)
            cv2.line(threshM2, (0, l), (threshM2.shape[1], l), 80)  # (direita, baixo)
        elif abs(histV_der_alt[l]) > 0 and abs(histV_der_alt[l + 1]) <= 0:
            print("lineV_der final", l)
            cv2.line(threshM2, (0, l), (threshM2.shape[1], l), 80)  # (direita, baixo)

    # linhas histograma cinzento claro
    for c in range(width):
        if abs(histH_alt[c]) > 0 and abs(histH_alt[c - 1]) <= 0:
            print("lineH inicial", c)
            cv2.line(threshM2, (c, 0), (c, threshM2.shape[0]), 180)  # (direita, baixo)
        elif abs(histH_alt[c]) > 0 and abs(histH_alt[c + 1]) <= 0:
            print("lineH final", c)
            cv2.line(threshM2, (c, 0), (c, threshM2.shape[0]), 180)  # (direita, baixo)

    cv2.imshow("M2", threshM2)
    # cv2.waitKey(0)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # pintar a preto

    # tirar apenas os maximos das derivadas para filtrar
    # linhas do histograma sem ser derivadas fornecem maximos
    # tamanhos das linhas considerando a área da massa
