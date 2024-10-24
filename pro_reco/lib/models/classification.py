from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import optuna

class Classification:
    def __init__(self, df, target, scoring, n_trials=10):
            self.df = df
            self.target = target
            self.n_trials = n_trials
            self.X = df.drop(target, axis=1)
            self.y = np.ravel(df[target])
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

    def optimizer(self, model_func):
        study = optuna.create_study(direction="maximize")
        study.optimize(model_func, n_trials=self.n_trials)
        return study

    def random_forest_model(self, trial):
        max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
        n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimator)
        return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

    def adaboost_model(self, trial):
        n_estimators = trial.suggest_int('n_estimators', self.start_n_estimator, self.end_n_estimator)
        learning_rate = trial.suggest_float('learning_rate', self.start_learning_rate, self.end_learning_rate)
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm='SAMME') 
        return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()
    
    def gradientboost_model(self, trial):
        max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
        n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
        model = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimator)
        return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

    def knn_model(self, trial):
        n_neighbors = trial.suggest_int('n_neighbors', self.start_n_neighbors, self.end_n_neighbors)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

    def catboost_model(self, trial):
        max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
        n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
        model = CatBoostClassifier(max_depth=max_depth, n_estimators=n_estimator)
        return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

    def xgboost_model(self, trial):
        max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
        n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
        model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimator)
        return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

    def gaussiannb_model(self, trial):
        var_smoothing = trial.suggest_int('var_smoothing', self.start_var_smoothing, self.end_var_smoothing)
        model = GaussianNB(var_smoothing=var_smoothing)
        return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

    def run_clf_models(self):
        scoring = self.scoring
        models = {
            'randomforest': self.optimizer(self.random_forest_model),
            'adaboost': self.optimizer(self.adaboost_model),
            'gradientboost': self.optimizer(self.gradientboost_model),
            'KNeighbors': self.optimizer(self.knn_model),
            'catboost': self.optimizer(self.catboost_model),
            'xgboost': self.optimizer(self.xgboost_model),
            'GaussianNB': self.optimizer(self.gaussiannb_model),
        }
        best_results = {model: study.best_value for model, study in models.items()} # Best score output
        # trial_results = {model: study.get_trials() for model, study in models.items()} # All trial score output
        trial_results = {model: [trial.value for trial in study.get_trials()] for model, study in models.items()}
       
        # return pd.DataFrame([results])

        best_results_df = pd.DataFrame.from_dict(best_results, orient='index', columns=[scoring])
        best_results_df.index = ['randomforest', 'adaboost', 'gradientboost', 'KNeighbors', 'catboost', 'xgboost', 'GaussianNB']
        trial_result_df = pd.DataFrame(trial_results)
        renamed_trial_df = trial_result_df.rename(columns=lambda x: x + '_' + scoring)
    
        return {'best': best_results_df.to_json(), 'trial': renamed_trial_df.to_json()}

def compare_clf_models(df, target, n_trials):
    '''
    train all Classification models
    1. accuracy
    2. recall
    3. precision
    4. f1_weighted
    '''
    scorings = ['accuracy', 'recall', 'precision', 'f1_weighted']
    results = dict()
    for idx, scoring in enumerate(scorings):
        results[idx] = Classification(df, target, scoring, n_trials).run_clf_models()
    return results


# class ClassificationModels:
#     class Datasets:
#         def __init__(self,df, target, models, n_trials=10):
#             # Data info
#             self.df = df
#             self.target = target
#             self.n_trials = n_trials
#             self.X = df.drop(target, axis=1)
#             self.y = np.ravel(df[target])
#             self.models = models
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

#     class RandomForestClf(Datasets):
#         def __init__(self, df, target, n_trials=50):
#             super().__init__(df, target, n_trials)

#         def model(self, trial):
#             max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
#             n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
#             model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimator)
#             return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
        
#         def run_model(self):
#             self.optimizer(self.model)
#             return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

#     class GradientBoostingClf(Datasets):
#         def __init__(self, df, target, n_trials=50):
#             super().__init__(df, target, n_trials)
            
#         def model(self, trial):
#             max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
#             n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
#             model = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimator)
#             return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
        
#         def run_model(self):
#             self.optimizer(self.model)
#             return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}
        
#     class XgbClf(Datasets):
#         def __init__(self, df, target, n_trials=50):
#             super().__init__(df, target, n_trials)
            
#         def model(self, trial):
#             max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
#             n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
#             model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimator)
#             return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
        
#         def run_model(self):
#             self.optimizer(self.model)
#             return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}
        
#     class CatBoostClf(Datasets):
#         def __init__(self, df, target, n_trials=50):
#             super().__init__(df, target, n_trials)
            
#         def model(self, trial):
#             max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
#             n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
#             # learning_rate = trial.suggest_int("learning_rate", 0.5, 1)
#             model = CatBoostClassifier(max_depth=max_depth, n_estimators=n_estimator) # , learning_rate=learning_rate
#             return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
        
#         def run_model(self):
#             self.optimizer(self.model)
#             return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

#     class AdaboostClf(Datasets):
#         def __init__(self, df, target, n_trials=10):
#             super().__init__(df, target, n_trials)

#         def model(self, trial):
#             n_estimators = trial.suggest_int('n_estimators', self.start_n_estimator, self.end_n_estimator)
#             learning_rate = trial.suggest_float('learning_rate', self.start_learning_rate, self.end_learning_rate)
#             # max_depth = trial.suggest_int('max_depth', 1, 10)
#             model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm='SAMME') # estimator=DecisionTreeClassifier(max_depth=max_depth),
            
#             return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
        
#         def run_model(self):
#             self.optimizer(self.model)
#             return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

#     class KNeighborsClf(Datasets):
#         def __init__(self, df, target, n_trials=10):
#             super().__init__(df, target, n_trials)

#         def model(self, trial):
#             n_neighbors = trial.suggest_int('n_neighbors', self.start_n_neighbors, self.end_n_neighbors)
#             model = KNeighborsClassifier(n_neighbors=n_neighbors)
#             return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
        
#         def run_model(self):
#             self.optimizer(self.model)
#             return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

#     class Gaussian(Datasets):
#         def __init__(self, df, target, n_trials=10):
#             super().__init__(df, target, n_trials)

#         def model(self, trial):
#             var_smoothing = trial.suggest_int('var_smoothing', self.start_var_smoothing, self.end_var_smoothing)
#             model = GaussianNB(var_smoothing=var_smoothing)
#             return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
        
#         def run_model(self):
#             self.optimizer(self.model)
#             return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

#     def __init__(self, df, target, n_trials=10):
#         self.random_clf_model = self.RandomForestClf(df, target, n_trials)
#         self.gradientboost_clf_model = self.GradientBoostingClf(df, target, n_trials)
#         self.xgboost_clf_model = self.XgbClf(df, target, n_trials)
#         self.catboost_clf_model = self.CatBoostClf(df, target, n_trials)
#         self.adaboost_clf_model = self.AdaboostClf(df, target, n_trials)
#         self.knn_clf_model = self.KNeighborsClf(df, target, n_trials)
#         self.gaussian_clf_model = self.Gaussian(df, target, n_trials)

#     def run_clf_models(self):
#         total_results = dict()
#         total_results['randomforest'] = self.random_clf_model.run_model()
#         total_results['gradient'] = self.gradientboost_clf_model.run_model()
#         total_results['xgboost'] = self.xgboost_clf_model.run_model()
#         total_results['catboost'] = self.catboost_clf_model.run_model()
#         total_results['adaboost'] = self.adaboost_clf_model.run_model()
#         total_results['knn'] = self.knn_clf_model.run_model()
#         total_results['gaussian'] = self.gaussian_clf_model.run_model()

#         # total_results[0] = self.random_clf_model.run_model()
#         # total_results[1] = self.gradientboost_clf_model.run_model()
#         # total_results[2] = self.xgboost_clf_model.run_model()
#         # total_results[3] = self.catboost_clf_model.run_model()
#         # total_results[4] = self.adaboost_clf_model.run_model()
#         # total_results[5] = self.knn_clf_model.run_model()
#         # total_results[6] = self.gaussian_clf_model.run_model()
#         return total_results