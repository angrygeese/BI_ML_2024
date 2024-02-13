import numpy as np
from itertools import product


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        distances = np.zeros((len(X), len(self.train_X)))
        for pos_test, vector_test in enumerate(X):
            for pos_train, vector_train in enumerate(self.train_X):
                dist = np.sum(np.abs(vector_test - vector_train))
                distances[pos_test, pos_train] = dist
        
        return distances

    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        """
        YOUR CODE IS HERE
        """
        distances = np.zeros((len(X), len(self.train_X)))
        for pos_test, vector_test in enumerate(X):
            test_X_expanded = np.full((len(self.train_X), len(vector_test)), vector_test)
            distances[pos_test] = np.sum(np.abs(test_X_expanded - self.train_X), axis=1)
        
        return distances


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
   
        X_test_column = X[:, None]  # трансформируем вектор-строку в вектор-столбец
        X_test_matrix = np.tile(X_test_column, (len(self.train_X), 1))  # скопируем вектор-столбец `len(self.train_X)` раз, получим матрицу
        X_train_matrix = np.tile(self.train_X, (len(X), 1, 1))  # повторяем вектор-строку `len(X)`, получим матрицу
        distances = np.sum(np.abs(X_test_matrix - X_train_matrix), axis=2)  # вычтем матрицы
        
        return distances


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        # n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)
        # prediction = np.empty(n_test, dtype=object)


        for test_pos, test_sample in enumerate(distances):
            ind_of_min = np.argpartition(test_sample, self.k)[:self.k]
            classes_of_min = self.train_y[ind_of_min]
            classes_unique, classes_counts = np.unique(classes_of_min, return_counts=True)
            class_prediction = classes_unique[np.argmax(classes_counts)]
            prediction[test_pos] = class_prediction
        return prediction


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        # n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, int)

        for test_pos, test_sample in enumerate(distances):
            ind_of_min = np.argpartition(test_sample, self.k)[:self.k]
            classes_of_min = self.train_y[ind_of_min]
            classes_unique, classes_counts = np.unique(classes_of_min, return_counts=True)
            class_prediction = classes_unique[np.argmax(classes_counts)]
            prediction[test_pos] = class_prediction
        return prediction
