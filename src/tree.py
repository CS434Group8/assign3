import numpy as np


class Node():
    """
    Node of decision tree

    Parameters:
    -----------
    prediction: int
            Class prediction at this node
    feature: int
            Index of feature used for splitting on
    split: int
            Categorical value for the threshold to split on for the feature
    left_tree: Node
            Left subtree
    right_tree: Node
            Right subtree
    """

    def __init__(self, prediction, feature, split, left_tree, right_tree):
        self.prediction = prediction
        self.feature = feature
        self.split = split
        self.left_tree = left_tree
        self.right_tree = right_tree


class DecisionTreeClassifier():
    """
    Decision Tree Classifier. Class for building the decision tree and making predictions

    Parameters:
    ------------
    max_depth: int
            The maximum depth to build the tree. Root is at depth 0, a single split makes depth 1 (decision stump)
    """

    def __init__(self, max_depth=None, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
    # take in features X and labels y
    # build a tree

    def fit(self, X, y):
        self.num_classes = len(set(y))
        self.root = self.build_tree(X, y, depth=1)

    # make prediction for each example of features X
    def predict(self, X):
        preds = [self._predict(example) for example in X]

        return preds

    # prediction for a given example
    # traverse tree by following splits at nodes
    def _predict(self, example):
        node = self.root
        while node.left_tree:
            if example[node.feature] < node.split:
                node = node.left_tree
            else:
                node = node.right_tree
        return node.prediction

    # accuracy

    def accuracy_score(self, X, y):
        preds = self.predict(X)
        accuracy = (preds == y).sum()/len(y)
        return accuracy

    # function to build a decision tree
    def build_tree(self, X, y, depth):

        num_samples, num_features = X.shape
        if(self.max_features != None):
            self.features_idx = np.random.randint(
                num_features, size=self.max_features)
        else:
            self.features_idx = np.arange(0, X.shape[1])
        # store data and information about best split
        # used when building subtrees recursively
        best_feature = None
        best_split = None
        best_gain = 0.0
        best_left_X = None
        best_left_y = None
        best_right_X = None
        best_right_y = None

        # what we would predict at this node if we had to
        # majority class
        num_samples_per_class = [np.sum(y == i)
                                 for i in range(self.num_classes)]
        prediction = np.argmax(num_samples_per_class)

        # if we haven't hit the maximum depth, keep building
        if depth <= self.max_depth:
            # consider each feature
            for feature in self.features_idx:
                # consider the set of all values for that feature to split on
                possible_splits = np.unique(X[:, feature])
                for split in possible_splits:
                    # get the gain and the data on each side of the split
                    # >= split goes on right, < goes on left
                    gain, left_X, right_X, left_y, right_y = self.check_split(
                        X, y, feature, split)
                    # if we have a better gain, use this split and keep track of data
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_split = split
                        best_left_X = left_X
                        best_right_X = right_X
                        best_left_y = left_y
                        best_right_y = right_y

        # if we haven't hit a leaf node
        # add subtrees recursively
        if best_gain > 0.0:
            left_tree = self.build_tree(
                best_left_X, best_left_y, depth=depth+1)
            right_tree = self.build_tree(
                best_right_X, best_right_y, depth=depth+1)
            return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

        # if we did hit a leaf node
        return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)

    # gets data corresponding to a split by using numpy indexing

    def check_split(self, X, y, feature, split):
        left_idx = np.where(X[:, feature] < split)
        right_idx = np.where(X[:, feature] >= split)
        left_X = X[left_idx]
        right_X = X[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]

        # calculate gini impurity and gain for y, left_y, right_y
        gain = self.calculate_gini_gain(y, left_y, right_y)
        return gain, left_X, right_X, left_y, right_y

    def calculate_gini_impurity(self, y):
        y_pos = (y == 1).sum()/len(y)
        y_neg = (y == 0).sum()/len(y)
        y_impurity = 1-y_pos**2-y_neg**2
        return y_impurity

    def calculate_gini_gain(self, y, left_y, right_y):
        # not a leaf node
        # calculate gini impurity and gain
        gain = 0
        if len(left_y) > 0 and len(right_y) > 0:
            y_impurity = self.calculate_gini_impurity(y)
            left_impurity = self.calculate_gini_impurity(left_y)
            right_impurity = self.calculate_gini_impurity(right_y)

            weight_left = len(left_y)/len(y)
            weight_right = len(right_y)/len(y)

            gain = y_impurity-left_impurity*weight_left-right_impurity*weight_right
            # print(gain)

            ########################################
            #       YOUR CODE GOES HERE            #
            ########################################

            return gain
        # we hit leaf node
        # don't have any gain, and don't want to divide by 0
        else:
            return 0


class RandomForestClassifier():
    """
    Random Forest Classifier. Build a forest of decision trees.
    Use this forest for ensemble predictions

    YOU WILL NEED TO MODIFY THE DECISION TREE VERY SLIGHTLY TO HANDLE FEATURE BAGGING

    Parameters:
    -----------
    n_trees: int
            Number of trees in forest/ensemble
    max_features: int
            Maximum number of features to consider for a split when feature bagging
    max_depth: int
            Maximum depth of any decision tree in forest/ensemble
    """

    def __init__(self, n_trees, max_features, max_depth):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.seed = None
        self.trees = []
        for i in range(0, self.n_trees):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, max_features=self.max_features)
            self.trees.append(tree)

        ##################
        # YOUR CODE HERE #
        ##################

    # fit all trees

    def fit(self, X, y):
        if(self.seed != None):
            print('seed: ', self.seed)
            np.random.seed(self.seed)

        bagged_X, bagged_y = self.bag_data(X, y)
        print('Fitting Random Forest...\n')
        for i in range(self.n_trees):
            print('Test Cases: ', i+1, end='\t\r')
            self.trees[i].fit(bagged_X[i], bagged_y[i])
            ##################
            # YOUR CODE HERE #
            ##################
        print()

    def bag_data(self, X, y, proportion=1.0):
        bagged_X = []
        bagged_y = []
        Length = len(X)
        # print(Length)
        for i in range(self.n_trees):
            treeX = []
            treeY = []
            for j in range(len(X)):
                randInt = np.random.randint(Length-1)
                treeX.append(X[randInt])
                treeY.append(y[randInt])
            bagged_X.append(treeX)
            bagged_y.append(treeY)
            ##################
            # YOUR CODE HERE #
            ##################
        bagged_X = np.array(bagged_X)
        bagged_y = np.array(bagged_y)
        # ensure data is still numpy arrays
        return bagged_X, bagged_y

    def predict(self, X):
        preds = np.zeros(len(X))
        miniVotes = self.n_trees/2

        for i in range(self.n_trees):
            Onepreds = np.array(self.trees[i].predict(X))
            preds = preds+Onepreds

        preds = preds.tolist()

        for i in range(len(preds)):
            if(preds[i] > miniVotes):
                preds[i] = 1
            else:
                preds[i] = 0

        # remove this one \/
        # preds = np.ones(len(X)).astype(int)
        # ^that line is only here so the code runs

        ##################
        # YOUR CODE HERE #
        ##################
        return preds


################################################
# YOUR CODE GOES IN ADABOOSTCLASSIFIER         #
# MUST MODIFY THIS EXISTING DECISION TREE CODE #
################################################

class StumpClassifier():

    def __init__(self):
        self.left_predic = 0
        self.right_predict = 0
        self.feature = None
        self.split = None

    def choosePrediction(self, weights, y, y_idx):
        positive_weight = 0
        negative_weight = 0
        index = 0
        for i in y_idx[0]:
            if(y[i] == 1):
                positive_weight += weights[i]
            else:
                negative_weight += weights[i]

        if(positive_weight > negative_weight):
            prediction = 1
        else:
            prediction = -1
        return prediction

    def amountOfSay(self, total_error):
        total_error+=0.001
        return (1/2) * np.log((1-total_error)/total_error)

    def fit(self, X, y, weights):
        possibleY = [-1, 1]
        num_samples, num_features = X.shape
        self.features_idx = np.arange(0, X.shape[1])
        # store data and information about best split
        # used when building subtrees recursively
        best_split = None
        lowest_WeightError = 1
        for feature in self.features_idx:
            # consider the set of all values for that feature to split on
            possible_splits = np.unique(X[:, feature])
            for split in possible_splits:
                # get the gain and the data on each side of the split
                # >= split goes on right, < goes on left
                weight_error, prediction_left, prediction_right = self.check_weight_error(
                    X, y, feature, split, weights)
                # if we have a better gain, use this split and keep track of data
                if weight_error < lowest_WeightError:
                    lowest_WeightError = weight_error
                    best_split = split
                    self.left_predic = prediction_left
                    self.right_predict = prediction_right
                    self.feature = feature

        self.split = best_split
        # print('predic ',self.left_predic, self.right_predict)
        if(self.split==None):
            return 0
        return self.amountOfSay(lowest_WeightError)

    def check_weight_error(self, X, y, feature, split, weights):
        left_idx = np.where(X[:, feature] < split)
        right_idx = np.where(X[:, feature] >= split)
        left_X = X[left_idx]
        right_X = X[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]
        if len(left_y) > 0 and len(right_y) > 0:
            prediction_left = self.choosePrediction(weights, y, left_idx)
            prediction_right = self.choosePrediction(weights, y, right_idx)
            total_error = 0
            for i in left_idx[0]:
                if(y[i] != prediction_left):
                    total_error += weights[i]

            for j in right_idx[0]:
                if(y[j] != prediction_right):
                    total_error += weights[j]
            return total_error, prediction_left, prediction_right
        else:
            return 1, 0, 0

    # make prediction for each example of features X

    def predict(self, X):
        preds = [self._predict(example) for example in X]

        return preds

    # prediction for a given example
    # traverse tree by following splits at nodes
    def _predict(self, example):
        if(self.split==None):
            return 1
        if example[self.feature] < self.split:
            return self.left_predic
        else:
            return self.right_predict
    # accuracy

    def accuracy_score(self, X, y):
        preds = self.predict(X)
        accuracy = (preds == y).sum()/len(y)
        return accuracy


class AdaBoostClassifier():
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = []
        self.says = []
        for i in range(0, self.n_trees):
            tree = StumpClassifier()
            self.trees.append(tree)

    def updateWeights(self, predicts, y, say):
        index = 0
        # print('say:', say)
        for (y1, y2) in zip(predicts, y):
            if(y1 == y2):
                self.weights[index] *= np.exp(-say)
            else:
                self.weights[index] *= np.exp(say)
            index += 1
        total_weights = self.weights.sum()
        for i in range(0, len(self.weights)):
            self.weights[i] /= total_weights

            
    def fit(self, X, y):
        length = len(X)
        self.weights = np.full(len(X), 1/len(X))
        for tree in self.trees:
            # print('weights: ',self.weights)
            say = round(tree.fit(X, y, self.weights),5)
            self.says.append(say)
            predicts = tree.predict(X)
            self.updateWeights(predicts, y, say)
           
        
            
    def predict(self, X):
        preds = np.zeros(len(X))

        for i in range(self.n_trees):
            Onepreds = np.array(self.trees[i].predict(X))*self.says[i]
            preds = preds+Onepreds
        
        preds = preds.tolist()

        for i in range(len(preds)):
            if(preds[i] > 0):
                preds[i] = 1
            else:
                preds[i] = -1
        return preds