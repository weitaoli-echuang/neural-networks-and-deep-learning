# coding=utf-8
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt

clip_region = {
    "start_pos": np.array([0, 0]),
    "end_pos": np.array([0, 0])
}

learn_region = []


def list_events():
    events = [i for i in dir(cv2) if 'EVENT' in i]
    print(events)


def on_mouse(event, x, y, flags, param):
    global clip_region
    org = param[0]
    if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
        print("row %d, col %d", y, x)
        clip_region['start_pos'] = np.array([x, y])
        cv2.circle(org, (x, y), 10, (255, 0, 0))
    elif event == cv2.EVENT_LBUTTONUP and (flags & cv2.EVENT_FLAG_SHIFTKEY):
        print("row %d, col %d", y, x)
        clip_region['end_pos'] = np.array([x, y])
        if (clip_region['start_pos'] - clip_region['end_pos']).any():
            cv2.rectangle(org, tuple(clip_region['start_pos']), tuple(
                clip_region['end_pos']), (0, 0, 255), 2)
            train_data(org)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_CTRLKEY):
        print("row %d, col %d", y, x)
        learn_region.append((x, y))
        cv2.circle(org, (x, y), 10, (0, 255, 0))


def on_learn():
    pass


def train_data(image):
    dim = np.abs(clip_region['start_pos'] - clip_region['end_pos'])
    x = min(clip_region['start_pos'][0], clip_region['end_pos'][0])
    y = min(clip_region['start_pos'][1], clip_region['end_pos'][1])

    img = image[y:y + dim[1], x:x + dim[0], :3]
    # cv2.namedWindow('clip')
    # cv2.imshow('clip', img)
    #
    kernal = np.array([16, 16])
    half_kernal = kernal / 2
    dim -= kernal
    tmp = np.random.rand(16).reshape(4, 4)
    _, axarr = plt.subplots(6, 6)
    for i in range(1, 6):
        for j in range(1, 6):
            axarr[i, j].imshow(tmp)
    plt.show()
    return img


def main():
    file_name = 'image/image000835small.png'

    image = cv2.imread(file_name, 1)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)

    x = tf.Variable(image, name='x')
    model = tf.global_variables_initializer()

    with tf.Session() as sess:
        x = tf.transpose(x, perm=[1, 0, 2])
        sess.run(model)
        result = sess.run(x)

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('image/image000835smallsmall.png')
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', on_mouse, [img])

    while 1:
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    # crop_img = img[crop_region[0]:crop_region[1], crop_region[2]:crop_region[3], :]
    # cv2.namedWindow('croped_image', cv2.WINDOW_NORMAL)
    # cv2.imshow('croped_image', crop_img)

    cv2.destroyAllWindows()
