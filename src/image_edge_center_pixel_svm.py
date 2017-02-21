"""
mnist_svm
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

# Libraries
# My libraries
#import mnist_loader
import load_pixels_block as pixel_load

# Third-party libraries
from sklearn import svm


def svm_baseline():
    #training_data, validation_data, test_data = mnist_loader.load_data()
    # train
    training_data, test_data = pixel_load.load_training_data_svm(
        'examples.txt', 120000, 3)
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    # test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print "Baseline classifier using an SVM."
    print "%s of %s values correct." % (num_correct, len(test_data[1]))

if __name__ == "__main__":
    svm_baseline()
