from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from abc import ABC, abstractmethod
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import optuna

class Classification:
    def __init__(self, df, target, scoring, n_trials=10):
        self.df = df
        self.target = target

        self.n_trials = n_trials
        self.X = df.drop(target, axis=1)
        self.y = np.ravel(df[target])
        # self.models = models
        self.scoring = scoring
        # tree based models params info
        self.start_n_estimator = 10
        self.end_n_estimator = 15
        self.start_max_depth = 6
        self.end_max_depth = 16

        self.start_learning_rate = 0.01
        self.end_learning_rate = 1

        self.start_n_neighbors = 5
        self.end_n_neighbors = 10

        self.start_var_smoothing = 1e-9
        self.end_var_smoothing = 1e-5

    def optimizer(self, objective):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        return study
    # def optimizer(self):
    #     self.study = optuna.create_study(direction="miximize")

    def random_forest_clf(self, trial):
        max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
        n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimator)
        return np.mean(cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring))
    
    def gradient_boosting_clf(self, trial):
        max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
        n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
        model = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimator)
        return np.mean(cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring))

    def xgb_clf(self, trial):
        max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
        n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
        model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimator)
        return np.mean(cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring))
    
    def catboost_clf(self, trial):
        max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
        n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
        model = CatBoostClassifier(max_depth=max_depth, n_estimators=n_estimator)
        return np.mean(cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring))

    def adaboost_clf(self, trial):
        n_estimators = trial.suggest_int('n_estimators', self.start_n_estimator, self.end_n_estimator)
        learning_rate = trial.suggest_float('learning_rate', self.start_learning_rate, self.end_learning_rate)
        # max_depth = trial.suggest_int('max_depth', 1, 10)
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm='SAMME') 
        return np.mean(cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring))

    def kneighbors_clf(self, trial):
        n_neighbors = trial.suggest_int('n_neighbors', self.start_n_neighbors, self.end_n_neighbors)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        return np.mean(cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring))
        
    def gaussian_clf(self, trial):
        var_smoothing = trial.suggest_int('var_smoothing', self.start_var_smoothing, self.end_var_smoothing)
        model = GaussianNB(var_smoothing=var_smoothing)
        return np.mean(cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring))

    def run(self):
        models = {
            'randomforest': self.optimizer(self.random_forest_clf),
            'gradientboost': self.optimizer(self.gradient_boosting_clf),
            'xgboost': self.optimizer(self.xgb_clf),
            'catboost': self.optimizer(self.catboost_clf),
            'adaboost': self.optimizer(self.adaboost_clf),
            'kneighbors': self.optimizer(self.kneighbors_clf),
            'gaussian': self.optimizer(self.gaussian_clf),
        }
        best_results = {model: study.best_value for model, study in models.items()} # Best score output
        trial_results = {model: study.get_trials() for model, study in models.items()} # All trial score output

        # return pd.DataFrame([results])
        best_results_df = pd.DataFrame.from_dict(best_results, orient='index', columns=[self.scoring])
        # best_results_df.index = ['randomforest', 'gradientboost', 'xgboost', 'catboost', 'adaboost', 'kneighbors', 'gaussian']
        trial_results_df = pd.DataFrame.from_dict(trial_results, orient='index', columns=[self.scoring])
        # trial_results_df.index = ['randomforest', 'gradientboost', 'xgboost', 'catboost', 'adaboost', 'kneighbors', 'gaussian']
        return {'best': best_results_df.to_json(), 'trial': trial_results_df.to_json()}


if __name__ == "__main__":
    path = './Test_02.csv'
    df = pd.read_csv(path)
    
    target = 'Potability' 
    # scoring = 'accuracy'

    scorings = ['accuracy', 'recall']

    all_results = []
    for scoring in scorings:   
        result = Classification(df, target, scoring, n_trials=5).run()
        all_results.append(result)
    final_results = pd.concat(all_results)
    final_results.to_csv('./clf_tune2.csv', index=False)
    # result1.to_csv('./clf_tune2.csv', index=False)