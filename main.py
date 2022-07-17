import cv2
import numpy as np
import os
# import math

import M1_contour_max_corners
import TR_calculus
import json_save


# parametros
THRESHOLD = 50
MORFOLOGIA = 5
AREA = 1000
DISTANCE_BETWEEN_POINTS = 0.69  # microns
WAFER_NUMBER = 11

data = []


ficheiros = os.listdir('DRIE_Parametric_Study/Wafer'+str(WAFER_NUMBER))

for i in range(len(ficheiros)):

    # read image, flip and resize
    original = cv2.imread('DRIE_Parametric_Study/Wafer'+str(WAFER_NUMBER)+"/"+ficheiros[i])
    # print(ficheiros[i])

    if WAFER_NUMBER < 10:
        W_Number = ficheiros[i][0:2]
    else:
        W_Number = ficheiros[i][0:3]
    W_Row = ficheiros[i][-11:-9]
    W_Trench_Number = ficheiros[i][-8:-4]

    print(W_Number, W_Row, W_Trench_Number)

    # the next lines were just to overcome images that it could not detect the lines and was returning erros,
    # so I made this quick fix
    """
    if W_Number == 'W11' and W_Row == 'C5' and W_Trench_Number == 'XY21':
    #     cv2.waitKey(0)
        continue
    if W_Number == 'W11' and W_Row == 'A1' and W_Trench_Number == 'XY02':
    #     cv2.waitKey(0)
        continue
    if W_Number == 'W11' and W_Row == 'D3' and W_Trench_Number == 'XY22':
    #     cv2.waitKey(0)
        continue
    if W_Number == 'W11' and W_Row == 'C5' and W_Trench_Number == 'XY55':
    #     cv2.waitKey(0)
        continue
    if W_Number == 'W11' and W_Row == 'C5' and W_Trench_Number == 'XY50':
    #     cv2.waitKey(0)
        continue
    """

    # original = cv2.resize(original, (0, 0), fx=0.5, fy=0.5)
    original = cv2.flip(original, 0)

    # create B&W image with threshold
    cinzentos = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    ret_, thresh = cv2.threshold(cinzentos, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((MORFOLOGIA, MORFOLOGIA), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # calculate rotation angle to make up for image displacement when taking the picture in the microscope
    media, angle = TR_calculus.image_rotation_angle(thresh)

    # rotate original image
    M = cv2.getRotationMatrix2D((thresh.shape[1]//2, 0), angle, 1)
    thresh = cv2.warpAffine(thresh, M, (thresh.shape[1]-1, 0))
    original = cv2.warpAffine(original, M, (thresh.shape[1]-1, 0))

    # update greyscale image with the make up rotation
    cinzentos = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    thresh1 = thresh.copy()

    # creates a rectangle to ignore the area above the wafer surface
    coord = np.array([[0, 0], [thresh.shape[1]-1, 0], [thresh.shape[1]-1, media], [0, media]], np.int32)
    cv2.fillPoly(thresh1, [coord], 0)  # 32
    cv2.line(thresh1, (0, int(media)), (thresh.shape[1]-1, int(media)), 0, 1)

    # final mask before extreme points detection
    ret, thresh1 = cv2.threshold(thresh1, THRESHOLD, 255, cv2.THRESH_BINARY)

    # show images after first processing unit
    # cv2.imshow("ORIGINAL CORRECTED ANGLE", original)
    # cv2.imshow("GRAYSCALE", cinzentos)
    # cv2.imshow("THRESH1_", thresh1)

    # calculates the four extreme of the wafer trench
    top_left, bottom_left, bottom_right, top_right = M1_contour_max_corners.contour_max_corners(thresh1, original)

    # if W_Number == 'W11' and W_Row == 'C5' and W_Trench_Number == 'XY21':
    #     cv2.waitKey(0)

    # calculates and draws the contour with the maximum area
    contornos, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    jan_contorno = original.copy()
    area = {}
    for j in range(len(contornos)):
        area[j] = cv2.contourArea(contornos[j])
    max_area_key = max(area, key=area.get)
    # print(max_area_key, area)
    cv2.drawContours(jan_contorno, contornos, max_area_key, (0, 255, 0), 1, 8)

    # calculates the contour point nearer the exrteme points
    contour_extreme_points_index, contour_extreme_points_coords = TR_calculus.contour_points_nearer_the_extreme_points(jan_contorno, contornos, max_area_key, top_left, bottom_left, bottom_right, top_right)
    print(contour_extreme_points_index)
    print(contour_extreme_points_coords)

    # creates the linear regression for each trench edge and returns the angle, the line equation and the error (...)?
    ll_angle, ll_m, ll_b, ll_mse_error = TR_calculus.linear_regrssion_with_mse_error(jan_contorno, contornos, max_area_key, contour_extreme_points_index["tl"], contour_extreme_points_index["bl"], 1)
    print("LEFT LINE:", ll_angle, ll_m, ll_b, ll_mse_error)

    bl_angle, bl_m, bl_b, bl_mse_error = TR_calculus.linear_regrssion_with_mse_error(jan_contorno, contornos, max_area_key, contour_extreme_points_index["bl"], contour_extreme_points_index["br"], 0)
    print("BOTTOM LINE:", bl_angle, bl_m, bl_b, bl_mse_error)

    rl_angle, rl_m, rl_b, rl_mse_error = TR_calculus.linear_regrssion_with_mse_error(jan_contorno, contornos, max_area_key, contour_extreme_points_index["br"], contour_extreme_points_index["tr"], 1)
    print("RIGHT LINE:", rl_angle, rl_m, rl_b, rl_mse_error)

    tl_angle, tl_m, tl_b, tl_mse_error = TR_calculus.linear_regrssion_with_mse_error(jan_contorno, contornos, max_area_key, contour_extreme_points_index["tr"], contour_extreme_points_index["end"], 0)
    print("TOP LINE:", tl_angle, tl_m, tl_b, tl_mse_error)

    # TO DO:
    # resolver bug: quando faço rotação da imagem as vezes sai para fora da imagem
    # baixar um pixel ou dois quando se cria o poligono para eliminar parte de cima ???
    # exportar para json

    left_size = contour_extreme_points_coords["bl"][1] - contour_extreme_points_coords["tl"][1]
    right_size = contour_extreme_points_coords["br"][1] - contour_extreme_points_coords["tr"][1]
    top_size = contour_extreme_points_coords["tr"][0] - contour_extreme_points_coords["tl"][0]
    bottom_size = contour_extreme_points_coords["br"][0] - contour_extreme_points_coords["bl"][0]

    print("left_size:", left_size, "right_size:", right_size)
    print("bottom_size:", bottom_size, "top_size:", top_size)

    bot_left = TR_calculus.intersection_points(jan_contorno, ll_m, ll_b, bl_m, bl_b, contour_extreme_points_index["tl"], contour_extreme_points_index["bl"], contornos, max_area_key)
    bot_right = TR_calculus.intersection_points(jan_contorno, rl_m, rl_b, bl_m, bl_b, contour_extreme_points_index["br"], contour_extreme_points_index["tr"], contornos, max_area_key)
    top_right = TR_calculus.intersection_points(jan_contorno, rl_m, rl_b, tl_m, tl_b, contour_extreme_points_index["br"], contour_extreme_points_index["tr"], contornos, max_area_key)
    top_left = TR_calculus.intersection_points(jan_contorno, ll_m, ll_b, tl_m, tl_b, contour_extreme_points_index["tl"], contour_extreme_points_index["bl"], contornos, max_area_key)

    trench_depth = TR_calculus.trench_depth(jan_contorno, contour_extreme_points_coords, contour_extreme_points_index, contornos[max_area_key])

    print("trench_depth:", trench_depth)

    entry = json_save.update_json_file(ficheiros[i], left_size, bottom_size, right_size, top_size, ll_angle, ll_m, ll_b, ll_mse_error, bl_angle, bl_m, bl_b, bl_mse_error, rl_angle, rl_m, rl_b, rl_mse_error, tl_angle, tl_m, tl_b, tl_mse_error, DISTANCE_BETWEEN_POINTS, bot_left, bot_right, top_right, top_left, trench_depth, W_Number, W_Row, W_Trench_Number)

    # cv2.imshow("janela contorno", jan_contorno)
    # cv2.imwrite("teste/"+ficheiros[i][0:-4]+".png", jan_contorno)
    data.append(entry)

    cv2.imwrite("DRIE_Parametric_Study/Wafer"+str(WAFER_NUMBER)+"_/"+ficheiros[i][0:-4]+".png", jan_contorno)

json_save.save_json_file(data, WAFER_NUMBER)
cv2.destroyAllWindows()
