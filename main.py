import cv2.cv2
from PIL import Image, ImageDraw

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from cv2 import cv2

import random


# конвертирование двоичной матрицы в картинку для OpenCV
def get_opencv_img(image):
    img = image.astype(np.uint8)
    ret, imag = cv2.threshold(img, 0.5, 255, 0)
    return imag


def get_area(image):
    kernel = np.ones((2, 2), dtype=np.uint8)
    contours, hir = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # opencv_image = cv2.dilate(image, kernel, iterations=1)
    contours_img = np.zeros((image.shape[1], image.shape[0]), dtype=np.uint8)
    contours_img = cv2.drawContours(contours_img, contours, -1, (255, 255, 255), 1)
    paint_cordinates = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for cnt in contours:
                if cv2.pointPolygonTest(cnt, (x, y), False) != -1 and image[y][x] == 255:
                    paint_cordinates.append((x, y))
    return contours_img, paint_cordinates


# Функция возвращающая маску предмета, рисует
def get_img():
    im = Image.new('1', (400, 400), color=0)
    draw = ImageDraw.Draw(im)
    for _ in range(random.randint(1, 4)):
        draw.polygon([random.randint(10, im.size[0]-10) for x in range(8)], fill=1)
    return np.array(im)


#
def get_route(painting_list):
    route = []
    # Точки обходятся сверху вниз, слева направо
    # Если в точке есть предмет (маска == 1)- красит
    for (y, x) in painting_list:
        route.append(
            {
                'x': x,
                'y': y,
                'status': 1
            }
        )
    return route


def count_n_plot(im_arr, rt, plot=True):
    # Проверка и визуализация результата
    sec_mask = Image.new('1', (400, 400), color=0)  # Создается второе изображение аналогичного размера
    draw = ImageDraw.Draw(sec_mask)
    distance = 0  # Обнуляется счетчик пройденной дистанции
    prev_coords = (rt[0]['x'], rt[0]['y'])  # Координаты руки в момент старта
    for step in route:
        # Для каждого шага по маршруту считается пройденная рукой дистанция
        distance += ((prev_coords[0] - step['x']) ** 2 + (prev_coords[1] - step['y']) ** 2) ** 0.5
        prev_coords = step['x'], step['y']
        # Если в маршруте указан статус 1 точка в текущих координатах закрашивается
        if step['status'] == 1:
            draw.line((prev_coords[1], prev_coords[0], step['y'], step['x']), fill=1)

    sec_arr = np.array(sec_mask)
    if plot:
        fig, axs = plt.subplots(figsize=(12, 9), ncols=3)
        axs[0].imshow(im_arr)
        axs[0].set_title('Изначальная маска')
        axs[1].imshow(sec_arr)
        axs[1].set_title('Закрашенная область')
        axs[2].imshow(im_arr ^ sec_arr)
        axs[2].set_title('Разница между областями')
        plt.show()

    return sec_arr, distance

# Генерируется изображение
im_arr = get_img()
opencv_img = get_opencv_img(im_arr)
count_img, painting_list = get_area(opencv_img)
# Получается маршрут
route = get_route(painting_list)
# # Выводятся первые точки маршрута
# print(route[:2])
# # Производится проход по маршруту
sec_arr, distance = count_n_plot(im_arr, route)


