# encoding utf8
import numpy as np
import random
import network
import load_pixels_block as pixel_load


def vectorized_result(j, vector_sz):
    e = np.zeros(vector_sz)
    e[j] = 1.0
    return e


def load_training_data(file_name, training_length):
    with open(file_name) as f:
        data = f.readlines()
        print 'line number: ', len(data)
        print 'raduis: ', data[0]
        category_dimension = (2, 1)
        pixels = []
        category = []
        data_2 = data[1:]
        random.shuffle(data_2)
        for pixels_block in data_2[:training_length]:
            L = [float(num) / 255. for num in pixels_block.split()[:-1]]
            pixels.append(np.array(L).reshape(675, 1))
            category.append(vectorized_result(
                int(pixels_block.split()[-1]), category_dimension))

        print 'training pixels block size: ', len(pixels)
        print 'training category size: ', len(category)
        training_data = zip(pixels, category)

        pixels = []
        category = []
        for pixels_block in data_2[training_length:-1]:
            L = [float(num) / 255. for num in pixels_block.split()[:-1]]
            pixels.append(np.array(L).reshape(675, 1))
            category.append(int(pixels_block.split()[-1]))

        print 'training pixels block size: ', len(pixels)
        print 'training category size: ', len(category)

        testing_data = zip(pixels, category)

    return training_data, testing_data


def load_data():
    file_name = '.\\examples\\examples_959736_flip+45degree.txt'
    training_data, testing_data = pixel_load.load_training_data_net(
        file_name, 0.5, 3, 2)
    return training_data, testing_data


def init_net(size):
    net = network.Network(size)
    return net


def sgd(net, training_data, testing_data, learing_rate):
    net.SGD(training_data, 10000, 10, learing_rate, test_data=testing_data)


def main():
    #    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    #    net = network.Network([784, 30, 10])
    #    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    #fileName = 'examples.txt'
    fileName = '.\\examples\\examples_959736_flip+45degree.txt'
    #training_data, testing_data = load_training_data(fileName, 30000)
    training_data, testing_data = pixel_load.load_training_data_net(
        fileName, 0.5, 3, 2)

    net = network.Network([675, 225, 75, 2])
    #net.SGD(training_data, 10000, 10, 0.008, test_data=testing_data)
    net.SGD(training_data, 10000, 10, 0.01, test_data=testing_data)


if __name__ == '__main__':
    main()
