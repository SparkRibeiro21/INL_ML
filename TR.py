


"""

# Regressao Linear com metodo do sklearn

left_line_points_x.reshape((-1, 1))

print(left_line_points)
# print(left_line_points_x)
# print(left_line_points_y)

regr = LinearRegression()
# Train the model using the training sets
regr.fit(left_line_points_x, left_line_points_y)

print(regr.score(left_line_points_x, left_line_points_y))

# The coefficients
print("Coefficient: \n", regr.coef_)
# The coefficients
print("Intercept: \n", regr.intercept_)

p1 = np.array((2, 1))
p2 = np.array((2, 1))

p1[1] = 0
p1[0] = float((regr.intercept_ - p1[1]) / -regr.coef_)

p2[1] = jan_contorno.shape[0]-1
p2[0] = float((regr.intercept_ - p2[1]) / -regr.coef_)

print(p1, p2)

# cv2.line(jan_contorno, p1, p2, (0, 255, 255), 1)
"""

"""
 print(len(contornos))
 for i in range(len(contornos)):
     area = cv2.contourArea(contornos[i])
     # print(len(contornos[i]), cv2.arcLength(contornos[i], True))
     # print('c:', contornos[i])
     print(area)
     if area > AREA:
         cv2.drawContours(jan_contorno, contornos, i, cor_contorno, 1, 8, hierarchy)
         # x, y, w, h = cv2.boundingRect(contornos[i])
         # cv2.rectangle(jan_contorno, (x, y), (x + w, y + h), (255, 0, 0), 1)
     # else:
         # cv2.drawContours(jan_contorno, contornos, i, (0, 0, 255), 1, 8, hierarchy)
         # x, y, w, h = cv2.boundingRect(contornos[i])
         # cv2.rectangle(jan_contorno, (x, y), (x + w, y + h), (255, 0, 0), 1)
 """

"""
import cv2
import numpy as np
import os
import math
import imutils

import M1_contour_max_corners
import M2_histograms
import M3_contour_slopes

# ficheiros = os.listdir('Celinov_Automatised_Char')
ficheiros = os.listdir('teste')

# parametros
THRESHOLD = 130
MORFOLOGIA = 5
AREA = 1500

# Fazer uma trend line nas verticais

for i in range(len(ficheiros)):
    original = cv2.imread('teste/'+ficheiros[i])
    # original=cv2.imread('teste/w25003xy55.tif')
    print(ficheiros[i])
    original = cv2.resize(original, (0, 0), fx=0.25, fy=0.25)
    original = cv2.flip(original, 0)
    # cv2.imshow("original", original)








    cinzentos = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # desfocada = cv2.GaussianBlur(cinzentos,(5,5), 0)
    # o threshold tem que estar depois da rotação!!!
    # o threshold funciona bem? (ver slide apresentação INL)

    ret, thresh = cv2.threshold(cinzentos, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("threshold_antes", thresh)

    kernel = np.ones((MORFOLOGIA, MORFOLOGIA), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # thresh = cv2.morphologyEx( thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("threshold_depois", thresh)

    # calcula coordenadas para eliminar espaço
    i = 0
    while thresh[i, 0] == 255:
        i += 1
    j = 0
    while thresh[j, thresh.shape[1]-1] == 255:
        j += 1
    i += 1
    j += 1
    media = (i+j)/2

    angle = 180*math.atan2(j-i, thresh.shape[1])/math.pi
    print('angle', angle)
    M = cv2.getRotationMatrix2D((thresh.shape[1]//2, 0), angle, 1)
    thresh = cv2.warpAffine(thresh, M, (thresh.shape[1]-1, 0))
    original = cv2.warpAffine(original, M, (thresh.shape[1]-1, 0))
    cv2.imshow("ORIGINAL CORRECTED ANGLE", original)
    # cv2.imshow("cinzentos", cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))

    thresh1 = thresh.copy()
    # cv2.imshow("THRESH1", thresh1)



    coord = np.array([[0, 0], [thresh.shape[1]-1, 0], [thresh.shape[1]-1, media], [0, media]], np.int32)
    # print(coord)
    cv2.fillPoly(thresh1, [coord], 0)  # 32
    cv2.line(thresh1, (0, int(media)), (thresh.shape[1]-1, int(media)), 0, 1)  # 127

    ####################################################################################################################

    # cv2.imshow("thresh1", thresh1)
    # cv2.imshow("real thresh", cv2.threshold(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), THRESHOLD, 255, cv2.THRESH_BIN
    ARY_INV)[1])
    # cv2.imshow("real cinzentos", cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))


    # cv2.imshow("THRESH1", thresh1)


    M1_contour_max_corners.contour_max_corners(thresh1, original)
    # M2_histograms.histograms(thresh1, original)
    # M3_contour_slopes.contour_slopes(thresh1, original)


    contornos, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    jan_contorno = original.copy()
    cor_contorno = (0, 255, 0)  # verde
    for i in range(len(contornos)):
        area = cv2.contourArea(contornos[i])
        # print(len(contornos[i]), cv2.arcLength(contornos[i], True))
        # print('c:', contornos[i])
        print(area)
        if area > AREA:
            cv2.drawContours(jan_contorno, contornos, i, cor_contorno, 1, 8, hierarchy)
            x, y, w, h = cv2.boundingRect(contornos[i])
            cv2.rectangle(jan_contorno, (x, y), (x + w, y + h), (255, 0, 0), 1)
        else:
            cv2.drawContours(jan_contorno, contornos, i, (0, 0, 255), 1, 8, hierarchy)
            x, y, w, h = cv2.boundingRect(contornos[i])
            cv2.rectangle(jan_contorno, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imshow("janela contorno", jan_contorno)

    cv2.waitKey(0)

cv2.destroyAllWindows()

"""
