from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import optuna

class RegressionModels:
    def __init__(self, df, target, scoring, n_trials=10):
        self.df = df
        self.target = target
        self.n_trials = n_trials
        self.X = df.drop(target, axis=1)
        self.y = np.ravel(df[target])
        self.scoring = scoring
        # Tree-based models params info
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

    def optimizer(self, model_func):
        study = optuna.create_study(direction="maximize")
        study.optimize(model_func, n_trials=self.n_trials)
        return study

    def random_forest_model(self, trial):
        max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
        n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
        model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimator)
        return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

    def gradient_boosting_model(self, trial):
        max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
        n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
        model = GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimator)
        return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

    def xgboost_model(self, trial):
        max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
        n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
        model = XGBRegressor(max_depth=max_depth, n_estimators=n_estimator)
        return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

    def catboost_model(self, trial):
        max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
        n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
        model = CatBoostRegressor(max_depth=max_depth, n_estimators=n_estimator, verbose=0)
        return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

    def adaboost_model(self, trial):
        n_estimators = trial.suggest_int('n_estimators', self.start_n_estimator, self.end_n_estimator)
        learning_rate = trial.suggest_float('learning_rate', self.start_learning_rate, self.end_learning_rate)
        model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
        return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

    def decision_tree_model(self, trial):
        max_depth = trial.suggest_int('max_depth', self.start_max_depth, self.end_max_depth)
        model = DecisionTreeRegressor(max_depth=max_depth)
        return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

    def knn_model(self, trial):
        n_neighbors = trial.suggest_int('n_neighbors', self.start_n_neighbors, self.end_n_neighbors)
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

    def run_reg_models(self):
        scoring = self.scoring
        models = {
            'randomforest': self.optimizer(self.random_forest_model),
            'gradientboost': self.optimizer(self.gradient_boosting_model),
            'xgboost': self.optimizer(self.xgboost_model),
            'catboost': self.optimizer(self.catboost_model),
            'adaboost': self.optimizer(self.adaboost_model),
            'DecisionTree': self.optimizer(self.decision_tree_model),
            'KNeighbors': self.optimizer(self.knn_model),
        }
        best_results = {model: study.best_value for model, study in models.items()} # Best score output
        # trial_results = {model: study.get_trials() for model, study in models.items()} # All trial score output
        trial_results = {model: [trial.value for trial in study.get_trials()] for model, study in models.items()}
       
        # return pd.DataFrame([results])

        best_results_df = pd.DataFrame.from_dict(best_results, orient='index', columns=[scoring])
        best_results_df.index = ['randomforest', 'gradientboost', 'xgboost', 'catboost', 'adaboost', 'DecisionTree', 'KNeighbors']
        trial_result_df = pd.DataFrame(trial_results)
        renamed_trial_df = trial_result_df.rename(columns=lambda x: x + '_' + scoring)
    
        return {'best': best_results_df.to_json(), 'trial': renamed_trial_df.to_json()}

def compare_reg_models(df, target, n_trials):
    '''
    train all regression models
    idx 1 is neg_mean_squared_error
    idx 2 is neg_mean_absolute_error 
    '''
    scorings = ['neg_mean_squared_error', 'neg_mean_absolute_error']
    results = dict()
    for idx, scoring in enumerate(scorings):
        results[idx] = RegressionModels(df, target, scoring, n_trials).run_reg_models()
    return results


    # def run_reg_models(self):
    #     total_results = dict()
    #     total_results[1] = self.optimizer(self.random_forest_model)
    #     total_results[2] = self.optimizer(self.gradient_boosting_model)
    #     total_results[3] = self.optimizer(self.xgboost_model)
    #     total_results[4] = self.optimizer(self.catboost_model)
    #     total_results[5] = self.optimizer(self.adaboost_model)
    #     total_results[6] = self.optimizer(self.decision_tree_model)
    #     total_results[7] = self.optimizer(self.knn_model)
    #     return total_results

    # def run_reg_models(self):
    #     models = [
    #         self.random_forest_model,
    #         self.gradient_boosting_model,
    #         self.xgboost_model,
    #         self.catboost_model,
    #         self.adaboost_model,
    #         self.decision_tree_model,
    #         self.knn_model
    #     ]
    #     total_results = {i + 1: self.optimizer(model) for i, model in enumerate(models)}
    #     return total_results


# class RegressionModels:
#     class Datasets:
#         def __init__(self, df, target, n_trials=10): # models,
#             # Data info
#             self.df = df
#             self.target = target
#             self.n_trials = n_trials
#             self.X = df.drop(target, axis=1)
#             # self.y = df[target]
#             self.y = np.ravel(df[target])
#             # self.models = models
#             # tree based models params info
#             self.start_n_estimator = 10
#             self.end_n_estimator = 15
#             self.start_max_depth = 6
#             self.end_max_depth = 16

#             self.start_learning_rate = 0.01
#             self.end_learning_rate = 1

#             self.start_n_neighbors = 5
#             self.end_n_neighbors = 10

#             self.start_var_smoothing = 1e-9
#             self.end_var_smoothing = 1e-5
#             # Define study
#             self.study = optuna.create_study(direction="maximize")
        
#         def optimizer(self, model):
#             study = self.study
#             study.optimize(model, n_trials=self.n_trials)

#     class RandomForestReg(Datasets):
#         def __init__(self, df, target, n_trials=50):
#             super().__init__(df, target, n_trials)

#         def model(self, trial):
#             max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
#             n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
#             model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimator)
#             return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()
        
#         def run_model(self):
#             self.optimizer(self.model)
#             return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

#     class GradientBoostingReg(Datasets):
#         def __init__(self, df, target, n_trials=50):
#             super().__init__(df, target, n_trials)
            
#         def model(self, trial):
#             max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
#             n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
#             model = GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimator)
#             return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()
        
#         def run_model(self):
#             self.optimizer(self.model)
#             return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

#     class XgbReg(Datasets):
#         def __init__(self, df, target, n_trials=50):
#             super().__init__(df, target, n_trials)
            
#         def model(self, trial):
#             max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
#             n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
#             model = XGBRegressor(max_depth=max_depth, n_estimators=n_estimator)
#             return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()
        
#         def run_model(self):
#             self.optimizer(self.model)
#             return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

#     class CatBoostReg(Datasets):
#         def __init__(self, df, target, n_trials=50):
#             super().__init__(df, target, n_trials)
            
#         def model(self, trial):
#             max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
#             n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
#             # learning_rate = trial.suggest_int("learning_rate", 0.5, 1)
#             model = CatBoostRegressor(max_depth=max_depth, n_estimators=n_estimator) # , learning_rate=learning_rate
#             return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()
        
#         def run_model(self):
#             self.optimizer(self.model)
#             return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

#     class AdaboostReg(Datasets):
#         def __init__(self, df, target, n_trials=10):
#             super().__init__(df, target, n_trials)

#         def model(self, trial):
#             n_estimators = trial.suggest_int('n_estimators', self.start_n_estimator, self.end_n_estimator)
#             learning_rate = trial.suggest_float('learning_rate', self.start_learning_rate, self.end_learning_rate)
#             # max_depth = trial.suggest_int('max_depth', 1, 10)
#             model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate) # estimator=DecisionTreeClassifier(max_depth=max_depth),
            
#             return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()
        
#         def run_model(self):
#             self.optimizer(self.model)
#             return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

#     class DecisionTreeReg(Datasets):
#         def __init__(self, df, target, n_trials=10):
#             super().__init__(df, target, n_trials)
        
#         def model(self, trial):
#             max_depth = trial.suggest_int('max_depth', self.start_max_depth, self.end_max_depth)
#             model = DecisionTreeRegressor(max_depth=max_depth)
#             return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()
        
#         def run_model(self):
#             self.optimizer(self.model)
#             return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

#     class KNeighborsReg(Datasets):
#         def __init__(self, df, target, n_trials=10):
#             super().__init__(df, target, n_trials)

#         def model(self, trial):
#             n_neighbors = trial.suggest_int('n_neighbors', self.start_n_neighbors, self.end_n_neighbors)
#             model = KNeighborsRegressor(n_neighbors=n_neighbors)
#             return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()
        
#         def run_model(self):
#             self.optimizer(self.model)
#             return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

#     def __init__(self, df, target, n_trials=10):
#         self.random_reg_model = self.RandomForestReg(df, target, n_trials)
#         self.gradientboost_reg_model = self.GradientBoostingReg(df, target, n_trials)
#         self.xgboost_reg_model = self.XgbReg(df, target, n_trials)
#         self.catboost_reg_model = self.CatBoostReg(df, target, n_trials)
#         self.adaboost_reg_model = self.AdaboostReg(df, target, n_trials)
#         self.decision_reg_model = self.DecisionTreeReg(df, target, n_trials)
#         self.knn_reg_model = self.KNeighborsReg(df, target, n_trials)


#     def run_reg_models(self):
#         total_results = dict()
#         total_results[1] = self.random_reg_model.run_model()
#         total_results[2] = self.gradientboost_reg_model.run_model()
#         total_results[3] = self.xgboost_reg_model.run_model()
#         total_results[4] = self.catboost_reg_model.run_model()
#         total_results[5] = self.adaboost_reg_model.run_model()
#         total_results[6] = self.decision_reg_model.run_model()
#         total_results[7] = self.knn_reg_model.run_model()

#         return total_results
