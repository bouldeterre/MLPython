
class DataSet:
    def __init__(self, _X_train, _X_test, _y_train, _y_test):
        self.X_train = _X_train
        self.y_train = _y_train
        self.X_test = _X_test
        self.y_test = _y_test

    def __str__(self):
        return f"Set Shape:\n X_train:{self.X_train.shape} y_train:{self.y_train.shape} \n \
X_test:{self.X_test.shape} y_test:{self.y_test.shape}"

