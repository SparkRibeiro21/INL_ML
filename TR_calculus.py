import math
import numpy as np
import cv2


def image_rotation_angle(thresh):  # to make up the erro angle the image is taken at

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
    # print('angle', angle)

    return media, angle


# calculates the contour point nearer the exrteme points
def contour_points_nearer_the_extreme_points(jan_contorno, contornos, max_area_key, top_left, bottom_left, bottom_right, top_right):
    # print(contornos[max_area_key])
    # print(contornos[max_area_key][0])
    # print(contornos[max_area_key][0][0][0])

    # print(len(contornos[max_area_key]))
    # print(type(contornos[max_area_key][0][0]))

    d_tl = np.zeros(len(contornos[max_area_key]))
    d_bl = np.zeros(len(contornos[max_area_key]))
    d_br = np.zeros(len(contornos[max_area_key]))
    d_tr = np.zeros(len(contornos[max_area_key]))

    for k in range(len(contornos[max_area_key])):
        # por cada ponto do contorno

        # cria vetor com distância a um dos mark points
        # d = sqrt((xA-xB)²+(yA-yB)²)

        # print(math.sqrt((contornos[max_area_key][k][0][0] - top_left[0])**2 + (contornos[max_area_key][k][0][1] - top_left[1])**2))
        d_tl[k] = math.sqrt((contornos[max_area_key][k][0][0] - top_left[0]) ** 2 + (
                    contornos[max_area_key][k][0][1] - top_left[1]) ** 2)
        d_bl[k] = math.sqrt((contornos[max_area_key][k][0][0] - bottom_left[0]) ** 2 + (
                    contornos[max_area_key][k][0][1] - bottom_left[1]) ** 2)
        d_br[k] = math.sqrt((contornos[max_area_key][k][0][0] - bottom_right[0]) ** 2 + (
                    contornos[max_area_key][k][0][1] - bottom_right[1]) ** 2)
        d_tr[k] = math.sqrt((contornos[max_area_key][k][0][0] - top_right[0]) ** 2 + (
                    contornos[max_area_key][k][0][1] - top_right[1]) ** 2)

    # print(min(d_tl))
    # print(np.argmin(d_tl))
    # print(min(d_bl))
    # print(np.argmin(d_bl))

    # print(min(d_br))
    # print(np.argmin(d_br))
    # print(min(d_tr))
    # print(np.argmin(d_tr))

    # start point for debug
    # cv2.circle(jan_contorno, contornos[max_area_key][np.argmin(d_tl)][0], 0, (255, 0, 0), 1)
    # cv2.circle(jan_contorno, contornos[max_area_key][np.argmin(d_bl)][0], 0, (255, 0, 0), 1)
    # cv2.circle(jan_contorno, contornos[max_area_key][np.argmin(d_br)][0], 0, (255, 0, 0), 1)
    # cv2.circle(jan_contorno, contornos[max_area_key][np.argmin(d_tr)][0], 0, (255, 0, 0), 1)

    d_br_inv = d_br[::-1]
    d_tr_inv = d_tr[::-1]
    cv2.circle(jan_contorno, contornos[max_area_key][len(contornos[max_area_key]) - np.argmin(d_br_inv) - 1][0], 0,
               (255, 255, 0), 1)
    cv2.circle(jan_contorno, contornos[max_area_key][len(contornos[max_area_key]) - np.argmin(d_tr_inv) - 1][0], 0,
               (255, 255, 0), 1)

    contour_extreme_points_index = {"tl": np.argmin(d_tl), "bl": np.argmin(d_bl),
                                    "br": len(contornos[max_area_key]) - np.argmin(d_br_inv) - 1,
                                    "tr": len(contornos[max_area_key]) - np.argmin(d_tr_inv) - 1,
                                    "end": len(contornos[max_area_key]) - 1}

    contour_extreme_points_coords = {"tl": contornos[max_area_key][np.argmin(d_tl)][0],
                                     "bl": contornos[max_area_key][np.argmin(d_bl)][0],
                                     "br": contornos[max_area_key][len(contornos[max_area_key]) - np.argmin(d_br_inv) - 1][0],
                                     "tr": contornos[max_area_key][len(contornos[max_area_key]) - np.argmin(d_tr_inv) - 1][0],
                                     "end": contornos[max_area_key][len(contornos[max_area_key]) - 1][0]}

    # print(contornos[max_area_key][np.argmin(d_tl)][0])
    # print(contornos[max_area_key][np.argmin(d_tl)][0][0])

    # contour_extreme_points = {"tl": 0, "bl": np.argmin(d_bl), "br": len(contornos[max_area_key])-np.argmin(d_br_inv)-1, "tr": len(contornos[max_area_key])-np.argmin(d_tr_inv)-1}

    # start point for debug
    cv2.circle(jan_contorno, contornos[max_area_key][np.argmin(d_tl)][0], 0, (255, 0, 0), 1)
    cv2.circle(jan_contorno, contornos[max_area_key][np.argmin(d_bl)][0], 0, (255, 0, 0), 1)
    cv2.circle(jan_contorno, contornos[max_area_key][np.argmin(d_br)][0], 0, (255, 0, 0), 1)
    cv2.circle(jan_contorno, contornos[max_area_key][np.argmin(d_tr)][0], 0, (255, 0, 0), 1)

    cv2.circle(jan_contorno, contornos[max_area_key][len(contornos[max_area_key]) - np.argmin(d_br_inv) - 1][0], 0, (255, 255, 0), 1)
    cv2.circle(jan_contorno, contornos[max_area_key][len(contornos[max_area_key]) - np.argmin(d_tr_inv) - 1][0], 0, (255, 255, 0), 1)

    print(len(contornos[max_area_key]))

    return contour_extreme_points_index, contour_extreme_points_coords


# creates the linear regression for each trench edge and returns the angle, the line equation and the error (...)?
def linear_regrssion_with_mse_error(jan_contorno, contornos, max_area_key, first_point, last_point, hor_ver):

    # if last_point - first_point + 1 > 0:
    num_points = last_point - first_point + 1

    line_points = np.zeros((num_points, 2))
    for a in range(first_point, last_point + 1):
        line_points[a - first_point] = contornos[max_area_key][a]

    # else:
    #     num_points = first_point - last_point + 1
    #
    #     line_points = np.zeros((num_points, 2))
    #     for a in range(last_point, first_point + 1):
    #         line_points[a - last_point] = contornos[max_area_key][a]

    p1 = np.array((2, 1))
    p2 = np.array((2, 1))

    [vx, vy, x, y] = cv2.fitLine(np.array(line_points), cv2.DIST_L2, 0, 0.01, 0.01)

    if hor_ver == 0:  # horizontal
        p1[0] = 0
        p1[1] = int((-x * vy / vx) + y)
        p2[0] = jan_contorno.shape[1] - 1
        p2[1] = int(((jan_contorno.shape[1] - 1 - x) * vy / vx) + y)

    if hor_ver == 1:  # vertical
        p1[0] = int((-y * vx / vy) + x)
        p1[1] = 0
        p2[0] = int(((jan_contorno.shape[0] - 1 - y) * vx / vy) + x)
        p2[1] = jan_contorno.shape[0] - 1

    cv2.line(jan_contorno, p1, p2, (0, 0, 255), 1)  # erro

    """
    p1[0] = 0
    p1[1] = int((-x * vy / vx) + y)
    p2[0] = jan_contorno.shape[0] - 1
    p2[1] = int(((jan_contorno.shape[0] - 1 - x) * vy / vx) + y)
    # cv2.line(jan_contorno, p1, p2, (0, 0, 255), 1)  # erro
    """

    if p1[0] == p2[0]:
        m = math.inf  # to solve: RuntimeWarning: divide by zero encountered in long_scalars -> m = (p1[1] - p2[1]) / (p1[0] - p2[0])
    else:
        m = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b = p1[1] - p1[0] * m
    angle = math.degrees(math.atan2(-(p1[1] - p2[1]), p1[0] - p2[0]))

    # print(angle)
    # normalize angle to fit the trigonometric circle

    # if angle < -135:
    #     angle += 180
    # elif angle > 135:  # negative angle if it is on the fourth quadrant example: -5º
    #     angle -= 180

    if angle < -135 or angle > 135:   # positive angle if it is on the fourth quadrant example: 355º
        angle += 180

    """

    if 45 < angle < 135:
        theta = angle - 90
        angle -= 2*theta
    elif angle < -45:
        angle = abs(angle)
    elif angle > 0:
        pass
    elif angle < 0:
        angle -= 180
        pass
    """

    # print("p1 =", p1, "p2 =", p2)
    # print("m =", m, "b =", b)

    """
    # print("(y=0)", -l_b / l_m, "(y=511)", (511 - l_b) / l_m)
    # if m != 0:  # caso o declive da reta seja 0, previne divisoes por 0
    #     print("(y=0)", int(-b / m), "(y=", jan_contorno.shape[0] - 1, ")", int((jan_contorno.shape[0] - 1 - b) / m))
    #     cv2.line(jan_contorno, (int(-b/m), 0), (int((jan_contorno.shape[0] - 1 - b) / m), jan_contorno.shape[0]), (0, 0, 255), 1)  # erro
    # else:
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("(y=0)", p1[1], "(y=", jan_contorno.shape[0] - 1, ")", p2[1])
    #     cv2.line(jan_contorno, (0, p1[1]), (jan_contorno.shape[1]-1, p2[1]), (0, 0, 255), 1)  # erro

    # if hor_ver == 0:  # horizontal
    #     pass
    #     cv2.line(jan_contorno, (0, p1[1]), (jan_contorno.shape[1]-1, p2[1]), (0, 0, 255), 1)  # erro
    # else:  # vertical
    #     cv2.line(jan_contorno, (int(-b/m), 0), (int((jan_contorno.shape[0] - 1 - b) / m), jan_contorno.shape[0]), (0, 0, 255), 1)  # erro
    #     pass
    """

    # calculate MSE (mean-square error):
    # 1) measure the distance of the observed y-values from the predicted y-values at each value of x;
    # 2) square each of these distances;
    # 3) calculate the mean of each of the squared distances.

    # percorrer os pontos ja isolados dos contornos left_line_points
    # calcular a distancia horizontal mesmo yy diferente xx
    # y = mx + b
    # x = (y-b)/m

    # since the lines are close to vertical and horizontal, it is more cost and time efficient to just check
    # the horizontal and vertical error respectively rather than calculating the distance to the trend line
    sum_error = 0
    if not math.isinf(m):
        for a in range(num_points):
            if hor_ver == 0:  # horizontal
                error = (line_points[a][1] - int(m * line_points[a][0] + b)) ** 2
                sum_error += error
                # print(a, line_points[a], int(m*line_points[a][0] + b + 1), error, sum_error)  # este é o valor do yy
            else:  # vertical
                error = (line_points[a][0] - int((line_points[a][1] - b) / m)) ** 2
                sum_error += error
                # print(a, line_points[a], int((line_points[a][1]-b)/m), error, sum_error)  # este é o valor do yy
        avg_error = sum_error / num_points
    else:  # math.isinf(m) # so acontece com linhas verticais em que o declive é 0
        for a in range(num_points):
            error = (line_points[a][0] - p1[0]) ** 2
            sum_error += error
            # print(a, line_points[a], p1, error, sum_error)  # este é o valor do yy
        avg_error = sum_error / num_points

    """
    # since the lines are close to vertical and horizontal, it is more cost and time efficient to just check
    # the horizontal and vertical error respectively rather than calculating the distance to the trend line
    sum_error = 0
    if hor_ver == 0:  # horizontal
        for a in range(num_points):
            error = (line_points[a][1] - int(m*line_points[a][0] + b + 1)) ** 2
            sum_error += error
            # print(a, line_points[a], int(m*line_points[a][0] + b + 1), error, sum_error)  # este é o valor do yy
        avg_error = sum_error / num_points
        # print(avg_error)
    else:  # vertical
        for a in range(num_points):
            error = (line_points[a][0] - int((line_points[a][1] - b) / m)) ** 2
            sum_error += error
            # print(a, line_points[a], int((line_points[a][1]-b)/m), error, sum_error)  # este é o valor do yy
        avg_error = sum_error / num_points
        # print(avg_error)

    # print(line_points)
    # for a in range(num_points):
    #     print(a, line_points[a], int((line_points[a][1]-b)/m), m*line_points[a][0] + b, m, b)
    """

    return angle, m, b, avg_error


def intersection_points(jan_contorno, m1, b1, m2, b2, first_point, last_point, contornos, max_area_key):

    # print(m1, b1, m2, b2)

    if math.isinf(m1):
        _sum_ = 0
        num_points = last_point - first_point + 1

        line_points = np.zeros((num_points, 2))
        for a in range(first_point, last_point + 1):
            line_points[a - first_point] = contornos[max_area_key][a]
            _sum_ += contornos[max_area_key][a][0][0]
            # print(contornos[max_area_key][a][0][0], _sum_)

        xi = _sum_ / num_points - 1
        # print(_sum_)
        yi = xi * m2 + b2

    else:

        xi = (b1 - b2) / (m2 - m1)
        yi = m1 * xi + b1

    # print('(xi,yi)', int(xi+0.5), int(yi+0.5))

    cc = (int(xi+0.5), int(yi+0.5))
    cv2.circle(jan_contorno, cc, 0, (45, 255, 255), 1)

    return cc


def trench_depth(jan_contorno, cepc, cepi, cont):

    # print(cepc)
    # print(cepc["bl"], cepc["br"])
    # print(cepc["tl"], cepc["tr"])

    p1 = int(cepc["bl"][0] + (cepc["br"][0] - cepc["bl"][0])/2 + 0.5)
    p2 = int((cepc["tl"][1] + cepc["tr"][1])/2 + 0.5)
    p2_ = 0

    # print(p1, p2)
    for i in range(cepi["br"]+1):
        if cont[i][0][0] == p1:
            # print(cont[i][0][0], cont[i][0][1])
            p2_ = cont[i][0][1]

    # cv2.circle(jan_contorno, (p1, p2), 0, (255, 0, 0), 1)
    # cv2.circle(jan_contorno, (p1, p2_), 0, (255, 0, 0), 1)

    cv2.line(jan_contorno, (p1, p2), (p1, p2_), (200, 0, 200), 1)  # erro

    return p2_-p2


