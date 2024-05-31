from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import optuna

class ClassificationModels:
    class Datasets:
        def __init__(self,df, target, models, n_trials=10):
            # Data info
            self.df = df
            self.target = target
            self.n_trials = n_trials
            self.X = df.drop(target, axis=1)
            self.y = df[target]
            self.models = models
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
            # Define study
            self.study = optuna.create_study(direction="maximize")
        
        def optimizer(self, model):
            study = self.study
            study.optimize(model, n_trials=self.n_trials)

    class RandomForestClf(Datasets):
        def __init__(self, df, target, n_trials=50):
            super().__init__(df, target, n_trials)

        def model(self, trial):
            max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
            n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
            model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimator, n_jobs=-1)
            return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
        
        def run_model(self):
            self.optimizer(self.model)
            return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

    class GradientBoostingClf(Datasets):
        def __init__(self, df, target, n_trials=50):
            super().__init__(df, target, n_trials)
            
        def model(self, trial):
            max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
            n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
            model = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimator)
            return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
        
        def run_model(self):
            self.optimizer(self.model)
            return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}
        
    class XgbClf(Datasets):
        def __init__(self, df, target, n_trials=50):
            super().__init__(df, target, n_trials)
            
        def model(self, trial):
            max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
            n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
            model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimator)
            return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
        
        def run_model(self):
            self.optimizer(self.model)
            return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}
        
    class CatBoostClf(Datasets):
        def __init__(self, df, target, n_trials=50):
            super().__init__(df, target, n_trials)
            
        def model(self, trial):
            max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
            n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
            # learning_rate = trial.suggest_int("learning_rate", 0.5, 1)
            model = CatBoostClassifier(max_depth=max_depth, n_estimators=n_estimator) # , learning_rate=learning_rate
            return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
        
        def run_model(self):
            self.optimizer(self.model)
            return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

    class AdaboostClf(Datasets):
        def __init__(self, df, target, n_trials=10):
            super().__init__(df, target, n_trials)

        def model(self, trial):
            n_estimators = trial.suggest_int('n_estimators', self.start_n_estimator, self.end_n_estimator)
            learning_rate = trial.suggest_float('learning_rate', self.start_learning_rate, self.end_learning_rate)
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
            n_neighbors = trial.suggest_int('n_neighbors', self.start_n_neighbors, self.end_n_neighbors)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
        
        def run_model(self):
            self.optimizer(self.model)
            return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

    class Gaussian(Datasets):
        def __init__(self, df, target, n_trials=10):
            super().__init__(df, target, n_trials)

        def model(self, trial):
            var_smoothing = trial.suggest_int('var_smoothing', self.start_var_smoothing, self.end_var_smoothing)
            model = GaussianNB(var_smoothing=var_smoothing)
            return cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').mean()
        
        def run_model(self):
            self.optimizer(self.model)
            return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

    def __init__(self, df, target, n_trials=10):
        self.random_clf_model = self.RandomForestClf(df, target, n_trials)
        self.gradientboost_clf_model = self.GradientBoostingClf(df, target, n_trials)
        self.xgboost_clf_model = self.XgbClf(df, target, n_trials)
        self.catboost_clf_model = self.CatBoostClf(df, target, n_trials)
        self.adaboost_clf_model = self.AdaboostClf(df, target, n_trials)
        self.knn_clf_model = self.KNeighborsClf(df, target, n_trials)
        self.gaussian_clf_model = self.Gaussian(df, target, n_trials)

    def run_clf_models(self):
        total_results = dict()
        total_results[0] = self.random_clf_model.run_model()
        total_results[1] = self.gradientboost_clf_model.run_model()
        total_results[2] = self.xgboost_clf_model.run_model()
        total_results[3] = self.catboost_clf_model.run_model()
        total_results[4] = self.adaboost_clf_model.run_model()
        total_results[5] = self.knn_clf_model.run_model()
        total_results[6] = self.gaussian_clf_model.run_model()


        return total_results




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