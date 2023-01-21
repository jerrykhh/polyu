import abc
import numpy as np
import math
import pandas as pd
import progressbar 

class Model(abc.ABC):
    
    def normailze(self, dataset):
        if isinstance(dataset, pd.DataFrame) or isinstance(dataset, pd.Series):
            return dataset.to_numpy()
        return dataset
    
    @abc.abstractmethod
    def predict(self, X):
        pass


class Node:
    
    def __init__(self, i_feature=None, threshold=None, value=None, left_tree=None, right_tree=None) -> None:
        self.i_feature = i_feature   # feature index
        self.threshold = threshold   # threshold for feature
        self.value = value           # class 
        self.left_tree = left_tree   # subtree (Left)
        self.right_tree = right_tree # subtree (Right)


class DecisionTree(Model):
    
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None  # Root node in dec. tree
        self.min_samples_split = min_samples_split   # Minimum n of samples to justify split
        self.min_impurity = min_impurity # The minimum impurity to justify split
        self.max_depth = max_depth   # maximum depth tree
        self.__impurity_cal_fuc = None  # Function for calculate impurity (info gain)
        self.__leaf_value_cal_func = None           # Function for predict y at leaf
    
    def set_impurity_cal_fuc(self, func):
        self.__impurity_cal_fuc = func

    
    def set_leaf_value_cal_func(self, func):
        self.__leaf_value_cal_func = func
    
    def fit(self, X, y):
        X = super().normailze(X)
        y = super().normailze(y)
        self.root = self.__build(X, y)
     
        
    def predict(self, X):
        X = super().normailze(X)
        return [self.__decison_value(sample) for sample in X]
            
    def __decison_value(self, x, tree=None):
        if tree is None:
            tree = self.root   
        
        # tree leaf
        if tree.value is not None:
          return tree.value
        
        feature = x[tree.i_feature]
        branch = tree.right_tree 
        
        if tree.threshold < feature:
            branch = tree.left_tree
        
        # get subtree decison
        return self.__decison_value(x, branch)
    
    
    def __divide_feature(self, X, i_feature, threshold):
        # Check sample value on the feature index is greater than a given threshold
        split_func = lambda sample: sample[i_feature] >= threshold if isinstance(threshold, int) or isinstance(threshold, float) else lambda sample: sample[i_feature] == threshold

        X_1 = np.array([sample for sample in X if split_func(sample)])
        X_2 = np.array([sample for sample in X if not split_func(sample)])
        
        return X_1, X_2
        
        
    def __build(self, X, y, depth=0):
        largest_impurity = 0
        best_criteria = None    # Feature index and threshold
        best_sets = None        # Subsets of the data
        
        y = np.expand_dims(y, axis=1) if len(np.shape(y)) == 1 else y
        # Put class col to last column of data(X)
        xy_data = np.concatenate((X, y), axis=1)
        
        n_samples, n_features = np.shape(X)
        
        if n_samples >= self.min_samples_split and depth <= self.max_depth:
            # Calculate impurity in each feature
            for feature_i in range(n_features):
                
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                
                for val in unique_values:
                    # Divide X and y to check whether the feature value of X at index feature_i satisfies a threshold
                    Xy_1, Xy_2 = self.__divide_feature(xy_data, feature_i, val)
                    if len(Xy_1) > 0 and len(Xy_2) > 0:
                        # Get y value to divided set
                        y1 = Xy_1[:, n_features:]
                        y2 = Xy_2[:, n_features:]
                        
                        # Calculate impurity
                        impurity = self.__impurity_cal_fuc(y, y1, y2)
                        
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": val}
                            best_sets = {
                                "left_x": Xy_1[:, :n_features],   # X of left subtree
                                "left_y": Xy_1[:, n_features:],   # y of left subtree
                                "right_x": Xy_2[:, :n_features],  # X of right subtree
                                "right_y": Xy_2[:, n_features:]   # y of right subtree
                            }
        
        if largest_impurity > self.min_impurity:
            # create subtrees
            left_branch = self.__build(best_sets["left_x"], best_sets["left_y"], depth + 1)
            right_branch = self.__build(best_sets["right_x"], best_sets["right_y"], depth + 1)
            
            # Root
            return Node(i_feature=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], left_tree=left_branch, right_tree=right_branch)
        
        # leaf value
        return Node(value=self.__leaf_value_cal_func(y))
    
                        
class ClassificationTree(DecisionTree):
    
    # calcuate entropy
    def __cal_entropy(self, y) -> float:
        unique_labels = np.unique(y)
        entropy = 0
        for class_label in unique_labels:
            count = len(y[ class_label == y])
            p = count / len(y)
            entropy += -p * math.log(p, 2)
        return entropy
        
    
    # calcuate infromation gain
    def __cal_inf_gain(self, y, y1, y2):
        # label entropy
        entropy = self.__cal_entropy(y)
        p = len(y1)/ len(y)
        inf_gain = entropy - p * self.__cal_entropy(y1) - (1-p) * self.__cal_entropy(y2)
        return inf_gain
    
    # majority vote
    def __vote(self, y):
        common_label = None
        max_count = 0
        for class_label in np.unique(y):
            count = len(y[y==class_label])
            if max_count < count:
                max_count = count
                common_label = class_label
        return common_label
    
    def fit(self, X, y):
        super().set_impurity_cal_fuc(self.__cal_inf_gain)
        super().set_leaf_value_cal_func(self.__vote)
        super().fit(X, y)
    
        
class RandomForest(Model):

    def __init__(self, n_estimators=100, min_samples_split=2,
                 min_gain=0, max_depth=float("inf")):
        self.n_estimators = n_estimators    # Number of classif trees
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain            # Minimum information gain req. to continue
        self.max_depth = max_depth          # Maximum depth for tree
        self.progress = progressbar.ProgressBar(widgets=['Tree Building: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="#", left="[", right="]"), ' ', progressbar.ETA()])
        
        # Initialize decision trees
        self.trees = [ClassificationTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_gain,
                    max_depth=self.max_depth) for _ in range(n_estimators)]


    def __get_random_subsets(self, X, y, n_subsets, replacements=True):
        n_samples, _ = np.shape(X)
        # Concatenate x and y and do a random shuffle
        X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
        np.random.shuffle(X_y)
        subsets = []

        # 50% of training samples
        subsample_size = int(n_samples // 2)
        if replacements:
            subsample_size = n_samples  

        for _ in range(n_subsets):
            idx = np.random.choice(
                range(n_samples),
                size=np.shape(range(subsample_size)),
                replace=replacements)
            X = X_y[idx][:, :-1]
            y = X_y[idx][:, -1]
            subsets.append([X, y])
        return subsets
    
    def fit(self, X, y):
        
        X = super().normailze(X)
        y = super().normailze(y)
        
        _, n_features = np.shape(X)
        # number of features to consider when looking for the best split
        self.max_features = int(math.sqrt(n_features))

        # Choose one random subset
        subsets = self.__get_random_subsets(X, y, self.n_estimators)

        for i in self.progress(range(self.n_estimators)):
            X_subset, y_subset = subsets[i]
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True) # choice random subsets of the features
            
            # Store the feature index to prediction
            self.trees[i].feature_indices = idx      
            X_subset = X_subset[:, idx] 
            # fit the data to subtree
            self.trees[i].fit(X_subset, y_subset)

    # Predirct
    def predict(self, X):
        X = super().normailze(X)
        y_preds = np.empty((X.shape[0], len(self.trees)))
        # prediction using builded classifcation tree
        for i, tree in enumerate(self.trees):
            # Feature index
            idx = tree.feature_indices
            # Pass the features to subtree to predict
            prediction = tree.predict(X[:, idx])
            y_preds[:, i] = prediction
            
        return [np.bincount(sample.astype('int')).argmax() for sample in y_preds]