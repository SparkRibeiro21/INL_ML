import cv2
import numpy as np
import math
import imutils

ROT_ANGLE = 45


def contour_max_corners(thresh1, original):

    # APLICAR A BORDA PARA RESOLVER BUG DE ROTACAO 45 EM QUE TRENCH SAI FORA DA IMAGEM

    # cv2.imshow("thresh_aux", thresh_aux)
    # imagem = cv2.imread('0.jpg', 0)
    # h, w = thresh_aux.shape[0:2]
    # print(h, w)
    # print(thresh_aux.shape)
    margem = 200
    wn = thresh1.shape[1] + 2 * margem
    hn = thresh1.shape[0] + 2 * margem
    thresh_aux = np.zeros((hn, wn), dtype=np.uint8)
    thresh_aux[margem:margem + thresh1.shape[0], margem:margem + thresh1.shape[1]] = thresh1

    # thresh3 = original.copy()

    # cv2.imshow("thresh2", thresh2)

    # RODA A IMAGEM PARA SE OBTER OS 4 EXTREMOS, TALVEZ OUTRO VALOR QUE NAO 45 PODE SER MELHOR
    M2 = cv2.getRotationMatrix2D((thresh_aux.shape[1]//2, thresh_aux.shape[0]//2), ROT_ANGLE, 1)
    thresh2 = cv2.warpAffine(thresh_aux, M2, (thresh_aux.shape[1], thresh_aux.shape[0]))

    # SE O CODIGO TIVER UMA TENDENCIA CLARA PARA UM DOS LADOS, PODE SOLUCIONAR EQUACIONAR -ROT_ANGLE
    # PENSO QUE O METODO COMO ESTA AGORA PODE ESTAR BIASED, POR UNS PIXEIS
    # M2 = cv2.getRotationMatrix2D((thresh1.shape[1]//2, thresh1.shape[0]//2), -ROT_ANGLE, 1)
    # thresh0 = cv2.warpAffine(thresh1, M2, (thresh1.shape[1], thresh1.shape[0]))

    # M3 = cv2.getRotationMatrix2D((thresh1.shape[1]//2, thresh1.shape[0]//2), ROT_ANGLE-10, 1)
    # thresh3 = cv2.warpAffine(thresh1, M3, (thresh1.shape[1], thresh1.shape[0]))

    ret2, thresh_aux = cv2.threshold(thresh_aux, 127, 255, cv2.THRESH_BINARY)  # THRESHOLD BINARY_INV
    # ret3, thresh3 = cv2.threshold(thresh3, 127, 255, cv2.THRESH_BINARY)  # THRESHOLD BINARY_INV

    thresh3 = cv2.flip(thresh2, 0)  # 0 = horizontal
    thresh4 = cv2.flip(thresh2, 1)  # 1 = vertical

    # contornos, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contornos, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print('contornos', list(contornos))

    # items = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contornos = items[0] if len(items) == 2 else items[1]

    # CONTORNOS A 45 GRAUS, SEM HIERARCHY PARA USAR FUNCAO IMUTILS
    contours2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours3 = cv2.findContours(thresh3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours4 = cv2.findContours(thresh4, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # print(contours)
    # contours = contours[0] if len(contours) == 2 else contours[1]

    # hierarchy = contours[1]

    contornos2 = imutils.grab_contours(contours2)
    c2 = max(contornos2, key=cv2.contourArea)

    contornos3 = imutils.grab_contours(contours3)
    c3 = max(contornos3, key=cv2.contourArea)

    contornos4 = imutils.grab_contours(contours4)
    c4 = max(contornos4, key=cv2.contourArea)

    # TUPLES DOS 4 EXTREMOS
    leftmost2 = tuple(c2[c2[:, :, 0].argmin()][0])
    rightmost2 = tuple(c2[c2[:, :, 0].argmax()][0])
    topmost2 = tuple(c2[c2[:, :, 1].argmin()][0])
    bottommost2 = tuple(c2[c2[:, :, 1].argmax()][0])

    # TUPLES DOS 2 EXTREMOS DO FLIP HORIZONTAL (LEFT E RIGHT)
    leftmost3 = tuple(c3[c3[:, :, 0].argmin()][0])
    rightmost3 = tuple(c3[c3[:, :, 0].argmax()][0])

    # TUPLES DOS 2 EXTREMOS DO FLIP VERTICAL (TOP E BOTTOM)
    topmost4 = tuple(c4[c4[:, :, 1].argmin()][0])
    bottommost4 = tuple(c4[c4[:, :, 1].argmax()][0])

    # CIRCULOS COM AS COORDENADAS DA IMAGEM A 45 GRAUS
    # cv2.drawContours(thresh2, [c], -1, 255, 2)
    cv2.circle(thresh2, leftmost2, 6, 128, 1)
    cv2.circle(thresh2, rightmost2, 6, 128, 1)

    cv2.circle(thresh2, topmost2, 6, 128, 1)
    cv2.circle(thresh2, bottommost2, 6, 128, 1)

    # COORDENADAS DO PONTO DA DIREITA
    # print('leftmost2:', leftmost2, 'rightmost2:', rightmost2)
    # print('leftmost3:', leftmost3, 'rightmost3:', rightmost3)

    # print('topmost2:', topmost2, 'bottommost2:', bottommost2)
    # print('topmost4:', topmost4, 'bottommost4:', bottommost4)

    # CIRCULOS COM AS COORDENADAS DA IMAGEM A -45 GRAUS
    cv2.circle(thresh3, leftmost3, 6, 128, 1)
    cv2.circle(thresh3, rightmost3, 6, 128, 1)

    cv2.circle(thresh4, topmost4, 6, 128, 1)
    cv2.circle(thresh4, bottommost4, 6, 128, 1)

    # COORDENADAS DO PONTO DA DIREITA
    # print(rightmost3)

    # LINHAS ENTRE OS 4 PONTOS
    # cv2.line(thresh2, leftmost, topmost, 127, 1)
    # cv2.line(thresh2, topmost, rightmost, 250, 1)
    # cv2.line(thresh2, rightmost, bottommost, 127, 1)
    # cv2.line(thresh2, bottommost, leftmost, 127, 1)

    # print(thresh3.shape[0]-1-rightmost3[1])
    corrigido = thresh3.shape[0]-1-rightmost3[1]
    y_ = int(0.5+(corrigido + rightmost2[1])/2)
    # print(y_)
    rightmost_corr = [rightmost2[0], y_]
    # print("rightmost_corr:", rightmost_corr)

    corrigido = thresh3.shape[0]-1-leftmost3[1]
    y_ = int(0.5+(corrigido + leftmost2[1])/2)
    # print(y_)
    leftmost_corr = [leftmost2[0], y_]
    # print("leftmost_corr:", leftmost_corr)

    corrigido = thresh4.shape[1]-1-topmost4[0]
    y_ = int(0.5+(corrigido + topmost2[0])/2)
    # print(y_)
    topmost_corr = [y_, topmost2[1]]
    # print("topmost_corr:", topmost_corr)

    corrigido = thresh4.shape[1]-1-bottommost4[0]
    y_ = int(0.5+(corrigido + bottommost2[0])/2)
    # print(y_)
    bottommost_corr = [y_, bottommost2[1]]
    # print("bottommost_corr:", bottommost_corr)

    cv2.circle(thresh2, rightmost_corr, 0, 100, 1)
    cv2.circle(thresh2, leftmost_corr, 0, 100, 1)

    cv2.circle(thresh2, topmost_corr, 0, 100, 1)
    cv2.circle(thresh2, bottommost_corr, 0, 100, 1)

    # cv2.imshow("thresh0", thresh0)
    # cv2.waitKey(0)

    # REROTAÇÃO DE -45 GRAUS PARA VOLTAR A POR A IMAGEM VERTICAL
    # M = cv2.getRotationMatrix2D((thresh2.shape[1]//2, thresh2.shape[0]//2), -ROT_ANGLE, 1)
    # thresh2 = cv2.warpAffine(thresh2, M, (thresh2.shape[1], thresh2.shape[0]))

    # cv2.transform(rightmost, M)
    # points = np.array(np.asarray(leftmost), np.asarray(rightmost), np.asarray(topmost), np.asarray(bottommost))

    # ones = np.ones(shape=(len(points), 1))
    # points_ones = np.hstack([points, ones])
    # transformed_point = M.dot(np.asarray(rightmost).T).T

    # print(thresh2[0])

    # CALCULOS PARA NOVAS COORDENADAS DOS 4 PONTOS DEPOIS DA REROTACAO
    # ORIGEM, PONTO E ANGULO
    ox = thresh2.shape[1]/2
    oy = thresh2.shape[0]/2
    angle = ROT_ANGLE

    px = rightmost_corr[0]
    py = rightmost_corr[1]
    rightmost_ = np.array([0, 0])
    rightmost_[0] = ox + math.cos(math.radians(angle)) * (px - ox) - math.sin(math.radians(angle)) * (py - oy) - margem
    rightmost_[1] = oy + math.sin(math.radians(angle)) * (px - ox) + math.cos(math.radians(angle)) * (py - oy) - margem

    px = leftmost_corr[0]
    py = leftmost_corr[1]
    leftmost_ = np.array([0, 0])
    leftmost_[0] = ox + math.cos(math.radians(angle)) * (px - ox) - math.sin(math.radians(angle)) * (py - oy) - margem - 1  # previne inicios sem ser no ponto 0 do contorno
    leftmost_[1] = oy + math.sin(math.radians(angle)) * (px - ox) + math.cos(math.radians(angle)) * (py - oy) - margem

    # FALTAM CORRECAO DE FLIP E MEDIA
    px = topmost_corr[0]
    py = topmost_corr[1]
    topmost_ = np.array([0, 0])
    topmost_[0] = ox + math.cos(math.radians(angle)) * (px - ox) - math.sin(math.radians(angle)) * (py - oy) - margem + 1  # compensa o outro ponot do topo que previne inicios sem ser no ponto 0 do contorno
    topmost_[1] = oy + math.sin(math.radians(angle)) * (px - ox) + math.cos(math.radians(angle)) * (py - oy) - margem

    # FALTAM CORRECAO DE FLIP E MEDIA
    px = bottommost_corr[0]
    py = bottommost_corr[1]
    bottommost_ = np.array([0, 0])
    bottommost_[0] = ox + math.cos(math.radians(angle)) * (px - ox) - math.sin(math.radians(angle)) * (py - oy) - margem
    bottommost_[1] = oy + math.sin(math.radians(angle)) * (px - ox) + math.cos(math.radians(angle)) * (py - oy) - margem

    # IMPRIMIR NOVAS COORDENADAS DO PONTO
    # print('leftmost_:', leftmost_, 'rightmost_:', rightmost_)
    # print('topmost_:', topmost_, 'bottommost_:', bottommost_)

    # DESENHA CIRCULO COM NOVAS COORDENADAS, PARA CONFIRMAR SE COINCIDE COM AS COORDENADAS ANTIGAS
    cv2.circle(original, rightmost_, 6, (0, 0, 255), 1)
    cv2.circle(original, rightmost_, 0, (0, 0, 255), 1)
    cv2.circle(original, leftmost_, 6, (0, 0, 255), 1)
    cv2.circle(original, leftmost_, 0, (0, 0, 255), 1)
    cv2.circle(original, topmost_, 6, (0, 0, 255), 1)
    cv2.circle(original, topmost_, 0, (0, 0, 255), 1)
    cv2.circle(original, bottommost_, 6, (0, 0, 255), 1)
    cv2.circle(original, bottommost_, 0, (0, 0, 255), 1)

    # cv2.imshow("thresh2", thresh2)
    # cv2.imshow("thresh3", thresh3)
    # cv2.imshow("thresh4", thresh4)
    # cv2.imshow("ORIGINAL CORRIGIDA", original)
    # cv2.waitKey(0)

    # LINHAS ENTRE OS 4 PONTOS
    # cv2.line(original, leftmost_, topmost_, (0, 0, 255), 1)
    # cv2.line(original, topmost_, rightmost_, (0, 0, 255), 1)
    # cv2.line(original, rightmost_, bottommost_, (0, 0, 255), 1)
    # cv2.line(original, bottommost_, leftmost_, (0, 0, 255), 1)

    # slope = (topmost[1] - rightmost[1]) / (topmost[0] - rightmost[0])
    # angle = math.degrees(math.atan(slope))
    # print('slope: ', slope, angle)

    # show the output image
    # cv2.imshow("Image", thresh2)
    # cv2.waitKey(0)

    # MOSTRA A IMAGEM
    # cv2.imshow("thresh2", thresh2)

    # cv2.imshow("thresh2", thresh2)
    # cv2.imshow("thresh3", thresh3)
    # cv2.imshow("thresh4", thresh4)
    # cv2.imshow("ORIGINAL CORRIGIDA", original)

    # cinzentos = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    return leftmost_, bottommost_, rightmost_, topmost_
