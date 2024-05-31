from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

from sklearn.linear_model import Ridge, Lasso, ElasticNet



from sklearn.model_selection import cross_val_score
import optuna

class RegressionModels:
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

    class RandomForestReg(Datasets):
        def __init__(self, df, target, n_trials=50):
            super().__init__(df, target, n_trials)

        def model(self, trial):
            max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
            n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
            model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimator)
            return cross_val_score(model, self.X, self.y, cv=5, scoring='neg_mean_squared_error').mean()
        
        def run_model(self):
            self.optimizer(self.model)
            return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

    class GradientBoostingReg(Datasets):
        def __init__(self, df, target, n_trials=50):
            super().__init__(df, target, n_trials)
            
        def model(self, trial):
            max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
            n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
            model = GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimator)
            return cross_val_score(model, self.X, self.y, cv=5, scoring='neg_mean_squared_error').mean()
        
        def run_model(self):
            self.optimizer(self.model)
            return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

    class XgbReg(Datasets):
        def __init__(self, df, target, n_trials=50):
            super().__init__(df, target, n_trials)
            
        def model(self, trial):
            max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
            n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
            model = XGBRegressor(max_depth=max_depth, n_estimators=n_estimator)
            return cross_val_score(model, self.X, self.y, cv=5, scoring='neg_mean_squared_error').mean()
        
        def run_model(self):
            self.optimizer(self.model)
            return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

    class CatBoostReg(Datasets):
        def __init__(self, df, target, n_trials=50):
            super().__init__(df, target, n_trials)
            
        def model(self, trial):
            max_depth = trial.suggest_int("max_depth", self.start_max_depth, self.end_max_depth)
            n_estimator = trial.suggest_int("n_estimator", self.start_n_estimator, self.end_n_estimator)
            # learning_rate = trial.suggest_int("learning_rate", 0.5, 1)
            model = CatBoostRegressor(max_depth=max_depth, n_estimators=n_estimator) # , learning_rate=learning_rate
            return cross_val_score(model, self.X, self.y, cv=5, scoring='neg_mean_squared_error').mean()
        
        def run_model(self):
            self.optimizer(self.model)
            return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

    class AdaboostReg(Datasets):
        def __init__(self, df, target, n_trials=10):
            super().__init__(df, target, n_trials)

        def model(self, trial):
            n_estimators = trial.suggest_int('n_estimators', self.start_n_estimator, self.end_n_estimator)
            learning_rate = trial.suggest_float('learning_rate', self.start_learning_rate, self.end_learning_rate)
            # max_depth = trial.suggest_int('max_depth', 1, 10)
            model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate) # estimator=DecisionTreeClassifier(max_depth=max_depth),
            
            return cross_val_score(model, self.X, self.y, cv=5, scoring='neg_mean_squared_error').mean()
        
        def run_model(self):
            self.optimizer(self.model)
            return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

    class DecisionTreeReg(Datasets):
        def __init__(self, df, target, models, n_trials=10):
            super().__init__(df, target, models, n_trials)
        
        def model(self, trial):
            max_depth = trial.suggest_int('max_depth', self.start_max_depth, self.end_max_depth)
            model = DecisionTreeRegressor(max_depth=max_depth)
            return cross_val_score(model, self.X, self.y, cv=5, scoring='neg_mean_squared_error').mean()
        
        def run_model(self):
            self.optimizer(self.model)
            return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

    class KNeighborsReg(Datasets):
        def __init__(self, df, target, n_trials=10):
            super().__init__(df, target, n_trials)

        def model(self, trial):
            n_neighbors = trial.suggest_int('n_neighbors', self.start_n_neighbors, self.end_n_neighbors)
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
            return cross_val_score(model, self.X, self.y, cv=5, scoring='neg_mean_squared_error').mean()
        
        def run_model(self):
            self.optimizer(self.model)
            return {'best_trial': self.study.best_trial, 'trials': self.study.get_trials()}

    def __init__(self, df, target, n_trials=10):
        self.random_reg_model = self.RandomForestReg(df, target, n_trials)
        self.gradientboost_reg_model = self.GradientBoostingReg(df, target, n_trials)
        self.xgboost_reg_model = self.XgbReg(df, target, n_trials)
        self.catboost_reg_model = self.CatBoostReg(df, target, n_trials)
        self.adaboost_reg_model = self.AdaboostReg(df, target, n_trials)
        self.decision_reg_model = self.DecisionTreeReg(df, target, n_trials)
        self.knn_reg_model = self.KNeighborsReg(df, target, n_trials)


    def run_reg_models(self):
        total_results = dict()
        total_results[1] = self.random_reg_model.run_model()
        total_results[2] = self.gradientboost_reg_model.run_model()
        total_results[3] = self.xgboost_reg_model.run_model()
        total_results[4] = self.catboost_reg_model.run_model()
        total_results[5] = self.adaboost_reg_model.run_model()
        total_results[6] = self.decision_reg_model.run_model()
        total_results[7] = self.knn_reg_model.run_model()

        return total_results
