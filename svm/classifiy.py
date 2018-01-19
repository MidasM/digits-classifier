from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # Number of test cases
    tc = 100

    # Load some digits (10 classes)
    # http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits
    digits = datasets.load_digits()

    # Create the train and test sets
    x_train, y_train = digits.data[:-tc], digits.target[:-tc]
    x_test, y_test = digits.data[-tc:], digits.target[-tc:]

    # Create the Support Vector Machine class
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    #   C is the penalty parameter
    #   gamma is the kernel coefficient
    clf = svm.SVC(C=1.0, gamma=0.01)

    # Fit the model given the training data
    clf.fit(x_train, y_train)

    # Predict the classes
    y_pred = clf.predict(x_test)

    # Get the accuracy
    print(accuracy_score(y_pred, y_test))

