
import numpy as np # library for numerical operations
import pandas as pd # library for data manipulation and analysis
from sklearn import tree # module for decision tree algorithms
from sklearn.model_selection import train_test_split # module for splitting data into training and testing sets
from sklearn.tree import DecisionTreeClassifier # class for creating a decision tree classifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import export_graphviz # function for exporting decision tree in DOT format
from sklearn.metrics import (
    accuracy_score,
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    average_precision_score
) # function for calculating accuracy score
from sklearn.metrics import confusion_matrix # function for calculating confusion matrix
from six import StringIO # class for working with strings as file-like objects
from IPython.display import Image # class for displaying images in IPython environment
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import EDA_PreP

class main:

    def get_score(y_true,y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)        

        print(f"Accuracy: {accuracy:.2f}") 
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1 Score:  {f1:.2f}")

    def getLogisticsReg():
        lr = LogisticRegression()
        return lr

    def getRandomForest(n_estimators,random_state):
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        return rf

    def getXgBoost(n_estimators, max_depth, learning_rate)
        xgb = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
        return xgb

    def getDecesionTree(criterion,max_depth)
        dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
        return dt

    def getLogisticsRegWithGridSearch(C,penalty,solver,class_weight,cv,scoring):
        param_grid = {
            'C': C,
            'penalty': penalty,
            'solver': solver
        }
        grid = GridSearchCV(
            LogisticRegression(class_weight=class_weight), 
            param_grid, 
            cv=cv, 
            scoring=scoring
        )
        return grid

    def getRandomForestWithGridSearch(n_estimators,max_depth,min_samples_leaf,class_weight,random_state,cv,scoring,n_jobs,verbose):
        param_grid = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': class_weight
        }
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=random_state),
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
        return grid_search

    def getXgBoostWithGridSearch(n_estimators, max_depth, learning_rate,scale_pos_weight,use_label_encoder,eval_metric,n_splits,shuffle,random_state,cv,scoring,n_jobs)
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
        param_grid = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'scale_pos_weight': scale_pos_weight # Test standard vs. theoretical balance
        }
        xgb_model = xgb.XGBClassifier(use_label_encoder=use_label_encoder, eval_metric=eval_metric)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )
        return grid_search

    def getDecesionTreeWithGridSearch(criterion,max_depth,min_samples_leaf,class_weight,random_state,cv,scoring,n_jobs)
        param_grid = {
            'criterion': criterion,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': class_weight 
        }
        grid_search = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=random_state),
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )
        return grid_search




