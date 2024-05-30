# ensemble
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, MeanShift, SpectralBiclustering
from sklearn.cluster import Birch, AgglomerativeClustering, SpectralClustering

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import optuna


class Datasets:
    def __init__(self,df, target, models, n_trials=10):
        self.df = df
        self.target = target
        self.n_trials = n_trials
        self.X = df.drop(target, axis=1)
        self.y = df[target]
        self.models = models
        self.study = optuna.create_study(direction="maximize")
    
    def optimizer(self, model):
        study = self.study
        study.optimize(model, n_trials=self.n_trials)

class RandomForestClf(Datasets):
    def __init__(self, df, target, n_trials=50):
        super().__init__(df, target, n_trials)

    def model(self, trial):
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=10)
        return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
    
    def run_model(self):
        self.optimizer(self.model)
        return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

class GradientBoostingClf(Datasets):
    def __init__(self, df, target, n_trials=50):
        super().__init__(df, target, n_trials)
        
    def model(self, trial):
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        model = GradientBoostingClassifier(max_depth=max_depth, n_estimators=10)
        return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
    
    def run_model(self):
        self.optimizer(self.model)
        return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}
    
class XgbClf(Datasets):
    def __init__(self, df, target, n_trials=50):
        super().__init__(df, target, n_trials)
        
    def model(self, trial):
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        model = XGBClassifier(max_depth=max_depth, n_estimators=10)
        return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
    
    def run_model(self):
        self.optimizer(self.model)
        return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}
    
class CatBoostClf(Datasets):
    def __init__(self, df, target, n_trials=50):
        super().__init__(df, target, n_trials)
        
    def model(self, trial):
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        model = CatBoostClassifier(max_depth=max_depth, n_estimators=10)
        return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
    
    def run_model(self):
        self.optimizer(self.model)
        return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

class AdaboostClf(Datasets):
    def __init__(self, df, target, n_trials=10):
        super().__init__(df, target, n_trials)

    def model(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0)
        # max_depth = trial.suggest_int('max_depth', 1, 10)
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm='SAMME') # estimator=DecisionTreeClassifier(max_depth=max_depth),
        
        return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
    
    def run_model(self):
        self.optimizer(self.model)
        return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

class KNeighborsClf(Datasets):
    def __init__(self, df, target, n_trials=10):
        super().__init__(df, target, n_trials)

    def model(self, trial):
        n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
    
    def run_model(self):
        self.optimizer(self.model)
        return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

class Gaussian(Datasets):
    def __init__(self, df, target, n_trials=10):
        super().__init__(df, target, n_trials)

    def model(self, trial):
        n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
        model = GaussianNB(n_neighbors=n_neighbors)
        
        return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
    
    def run_model(self):
        self.optimizer(self.model)
        return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}


class ClassificationModels:
    def __init__(self, df, target, n_trials=10):
        self.df = df
        self.target = target
        self.n_trials = n_trials 

    def ml_models(self):
        # datasets = {'df': self.df, 'target': self.target, 'n_trials': self.n_trials}
        randomforest_clf_model = RandomForestClf(self.df, self.target, n_trials=10)
        # randomforest_clf_model = RandomForestClf(**datasets)
        # gradientboost_clf_model = GradientBoostingClf(**datasets)
        # xgboost_clf_model = XgbClf(**datasets)
        # catboost_clf_model = CatBoostClf(**datasets)

        total_result = dict()        
        randomforest_clf_result = randomforest_clf_model.run_model()
        # gradientboost_clf_result = gradientboost_clf_model.run_model()
        # xgboost_clf_result = xgboost_clf_model.run_model()
        # catboost_clf_result = catboost_clf_model.run_model()
        
        total_result['1'] = randomforest_clf_result
        # total_result['2'] = gradientboost_clf_result
        # total_result['3'] = xgboost_clf_result
        # total_result['4'] = catboost_clf_result

        return total_result




# class Classification(Datasets):
#     def __init__(self, df, target, models, n_trials=10):
#         super().__init__(df, target, models, n_trials)
        
#     def model(self, trial):     
#         # learning_rate, max_depth, 
#         # n_estimators=10

#         # models = ['RandomForestClassifier' ,'AdaBoostClassifier', 'GradientBoostingClassifier' , 'KNeighborsClassifier', 'XGBClassifier', 
#         #           'GaussianNB', 'QuadraticDiscriminantAnalysis', 'LinearDiscriminantAnalysis', 'CatBoostClassifier']

#         # ['RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier', 'CatBoostClassifier', 'AdaBoostClassifier', 'KNeighborsClassifier', 'GaussianNB', 'QuadraticDiscriminantAnalysis']
#         models = self.models
#         model = None
#         if models == 'RandomForestClassifier':
#             max_depth = {'max_depth': trial.suggest_int("max_depth", 2, 32, log=True)}
#             model = RandomForestClassifier(**max_depth)

#         if models == 'GradientBoostingClassifier':
#             max_depth = {'max_depth': trial.suggest_int("max_depth", 2, 32, log=True)}
#             models = GradientBoostingClassifier(**max_depth)
        
#         if models == 'XGBClassifier':
#             max_depth = {'max_depth': trial.suggest_int("max_depth", 2, 32, log=True)}
#             model = XGBClassifier(**max_depth)
        
#         if models == 'CatBoostClassifier':
#             max_depth = {'max_depth': trial.suggest_int("max_depth", 2, 32, log=True)}
#             model = CatBoostClassifier(**max_depth)
        
#         if models == 'AdaBoostClassifier': # learning_rate, n_estimators
#             learning_rate = {'learning_rate': trial.suggest_int("learning_rate", 1, 10), 'algorithm': 'SAMME'}
#             model = AdaBoostClassifier(**learning_rate)
        
#         if models == 'KNeighborsClassifier':
#             n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
#             model = KNeighborsClassifier(**n_neighbors)
        
#         if models == 'GaussianNB':
#             var_smoothing = trial.suggest_loguniform('var_smoothing', 1e-9, 1e-3)
#             model = GaussianNB(**var_smoothing)
        
#         if models == 'QuadraticDiscriminantAnalysis':
#             reg_param = trial.suggest_uniform('reg_param', 0.0, 1.0)
#             model = QuadraticDiscriminantAnalysis(**reg_param)
#         # if models == 'LinearDiscriminantAnalysis':
#         #     model = LinearDiscriminantAnalysis()
        

#         return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
    
#     def run_model(self):
#         self.optimizer(self.model)
#         return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}