# encoding utf8
import numpy as np
import pdb


def vectorized_result(j, vector_sz):
    e = np.zeros((vector_sz, 1))
    e[j] = 1.0
    return e


def pixel_normal(pixel):
    return pixel / 255.


def load_data(fileName, channels):
    with open(fileName) as f:
        data = f.readlines()
        print 'line number: ', len(data)
        print 'raduis: ', data[0]
        pixels = []
        category = []
        array_sz = pow(2 * int(data[0].split()[0]) + 1, 2) * channels
        print array_sz
        stop_location = int(len(data) * 0.95)
        for pixels_block in data[1:stop_location]:
            L = [int(num) for num in pixels_block.split()]
            pixels.append(np.array(L[:-1]).reshape(array_sz, 1))
            category.append(L[-1])

        print 'training pixels block size: ', len(pixels)
        print 'training category size: ', len(category)

    return pixels, category


def load_training_data_net(fileName, training_rate, channels, labels_num):
    pixels, labels = load_data(fileName, channels)
    training_length = int(len(pixels) * training_rate)
    normal_pixels = [pixel_normal(pixel) for pixel in pixels]
    category = [vectorized_result(label, labels_num)
                for label in labels[:training_length]]
    training_data = zip(normal_pixels[:training_length], category)
    testing_data = zip(normal_pixels[training_length:], labels[
        training_length:])
    return training_data, testing_data


def load_training_data_svm(file_name, training_length, channels):
    pixels, labels = load_data(file_name, channels)
    row_len = len(pixels)
    col_len = len(pixels[0])
    normal_pixels = np.array([pixel_normal(pixel)
                              for pixel in pixels]).reshape(row_len, col_len)
    label_array = np.array(labels)

    training_data = (normal_pixels[:training_length],
                     label_array[:training_length])
    testing_data = (normal_pixels[training_length:],
                    label_array[training_length:])
    return training_data, testing_data
