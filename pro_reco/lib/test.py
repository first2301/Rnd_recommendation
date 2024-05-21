# ensemble
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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

# class RegressionCurv:
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def xgb(self):
#         X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
#         X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
#         train_errors, test_errors, val_errors = [], [], []
#         for m in range(1, len(X_train)):
#             xgb = XGBRegressor().fit(X_train[:m], y_train[:m])
#             xgb_train_pred = xgb.predict(X_train[:m])
#             xgb_test_pred = xgb.predict(X_test)
#             xgb_val_pred = xgb.predict(X_val)

#             train_errors.append(mean_absolute_error(y_train[:m], xgb_train_pred))
#             test_errors.append(mean_absolute_error(y_test, xgb_test_pred))
#             val_errors.append(mean_absolute_error(y_val, xgb_val_pred))

#         return {'train_errors': np.sqrt(train_errors), 'test_errors': np.sqrt(test_errors), 'val_errors': np.sqrt(val_errors)}
    
#     def rfc(self):
#         X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
#         X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
#         train_errors, test_errors, val_errors = [], [], []
#         for m in range(1, len(X_train)):
#             rfc = RandomForestRegressor().fit(X_train[:m], y_train[:m])
#             rfc_train_pred = rfc.predict(X_train[:m])
#             rfc_test_pred = rfc.predict(X_test)
#             rfc_val_pred = rfc.predict(X_val)

#             train_errors.append(mean_absolute_error(y_train[:m], rfc_train_pred))
#             test_errors.append(mean_absolute_error(y_test, rfc_test_pred))
#             val_errors.append(mean_absolute_error(y_val, rfc_val_pred))

#         return {'train_errors': np.sqrt(train_errors), 'test_errors': np.sqrt(test_errors), 'val_errors': np.sqrt(val_errors)}

#     def train_reg(self):
#         xgb_result = self.xgb()
#         rfc_result = self.rfc()
#         df = pd.DataFrame(xgb_result, rfc_result)

#         return df


PATH = 'F:\industry\data\clf_data\Test_02.csv'
df = pd.read_csv(PATH)
X = df.drop('Potability', axis=1)
y = df['Potability']

# result = RegressionCurv(X, y).train_reg()
# print(result)
import optuna
def objective(trial):
    rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
    classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=10)
    score = cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
result = study.trials



print(result)




class Classification:
    def __init__(self, df, target, n_trials=100):
        self.df = df
        self.target = target
        self.n_trials = n_trials
        self.study = optuna.create_study(direction="maximize")


    def train_test_set(self):
        df = self.df
        target = self.target
        X = df.drop(target, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def train_val_set(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val
    
    def accuracy_precision_recall_fscore(self, y_true, y_pred):
        accuracy, precision, recall, fscore = accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, average='weighted'), recall_score(y_true, y_pred, average='weighted'), f1_score(y_true, y_pred, average='weighted') 
        return accuracy, precision, recall, fscore

    def rfc_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = RandomForestClassifier().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_accuracy, train_precision, train_recall, train_fscore = self.accuracy_precision_recall_fscore(y_train, train_pred)
        test_accuracy, test_precision, test_recall, test_fscore = self.accuracy_precision_recall_fscore(y_test, test_pred)
        val_accuracy, val_precision, val_recall, val_fscore = self.accuracy_precision_recall_fscore(y_val, val_pred)
        df = pd.DataFrame({
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,

            'train_precision': train_precision,
            'val_precision': val_precision,
            'test_precision': test_precision,
            
            'train_recall': train_recall,
            'val_recall': val_recall,
            'test_recall': test_recall,
            
            'train_fscore': train_fscore,
            'val_fscore': val_fscore,
            'test_fscore': test_fscore,
            }, 
            index=['RandomForestClassifier'])
        
        return df
    
    def grad_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = GradientBoostingClassifier().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_accuracy, train_precision, train_recall, train_fscore = self.accuracy_precision_recall_fscore(y_train, train_pred)
        test_accuracy, test_precision, test_recall, test_fscore = self.accuracy_precision_recall_fscore(y_test, test_pred)
        val_accuracy, val_precision, val_recall, val_fscore = self.accuracy_precision_recall_fscore(y_val, val_pred)
        df = pd.DataFrame({
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'val_accuracy': val_accuracy,

            'train_precision': train_precision,
            'val_precision': val_precision,
            'test_precision': test_precision,
            
            'train_recall': train_recall,
            'val_recall': val_recall,
            'test_recall': test_recall,
            
            'train_fscore': train_fscore,
            'val_fscore': val_fscore,
            'test_fscore': test_fscore,
            }, 
            index=['GradientBoostingClassifier'])
        return df
    
    def xgb_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = XGBClassifier().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_accuracy, train_precision, train_recall, train_fscore = self.accuracy_precision_recall_fscore(y_train, train_pred)
        test_accuracy, test_precision, test_recall, test_fscore = self.accuracy_precision_recall_fscore(y_test, test_pred)
        val_accuracy, val_precision, val_recall, val_fscore = self.accuracy_precision_recall_fscore(y_val, val_pred)
        df = pd.DataFrame({
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,

            'train_precision': train_precision,
            'val_precision': val_precision,
            'test_precision': test_precision,
            
            'train_recall': train_recall,
            'val_recall': val_recall,
            'test_recall': test_recall,
            
            'train_fscore': train_fscore,
            'val_fscore': val_fscore,
            'test_fscore': test_fscore,
            }, 
            index=['XGBClassifier'])
        return df

    def knn_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = KNeighborsClassifier().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_accuracy, train_precision, train_recall, train_fscore = self.accuracy_precision_recall_fscore(y_train, train_pred)
        test_accuracy, test_precision, test_recall, test_fscore = self.accuracy_precision_recall_fscore(y_test, test_pred)
        val_accuracy, val_precision, val_recall, val_fscore = self.accuracy_precision_recall_fscore(y_val, val_pred)
        df = pd.DataFrame({
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,

            'train_precision': train_precision,
            'val_precision': val_precision,
            'test_precision': test_precision,
            
            'train_recall': train_recall,
            'val_recall': val_recall,
            'test_recall': test_recall,
            
            'train_fscore': train_fscore,
            'val_fscore': val_fscore,
            'test_fscore': test_fscore,
            }, 
            index=['KNeighborsClassifier'])
        return df

    def ada_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = AdaBoostClassifier().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_accuracy, train_precision, train_recall, train_fscore = self.accuracy_precision_recall_fscore(y_train, train_pred)
        test_accuracy, test_precision, test_recall, test_fscore = self.accuracy_precision_recall_fscore(y_test, test_pred)
        val_accuracy, val_precision, val_recall, val_fscore = self.accuracy_precision_recall_fscore(y_val, val_pred)
        df = pd.DataFrame({
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,

            'train_precision': train_precision,
            'val_precision': val_precision,
            'test_precision': test_precision,
        
            
            'train_recall': train_recall,
            'val_recall': val_recall,
            'test_recall': test_recall,

            
            'train_fscore': train_fscore,
            'val_fscore': val_fscore,
            'test_fscore': test_fscore,

            }, 
            index=['AdaBoostClassifier'])
        return df

    def gau_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = GaussianNB().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_accuracy, train_precision, train_recall, train_fscore = self.accuracy_precision_recall_fscore(y_train, train_pred)
        test_accuracy, test_precision, test_recall, test_fscore = self.accuracy_precision_recall_fscore(y_test, test_pred)
        val_accuracy, val_precision, val_recall, val_fscore = self.accuracy_precision_recall_fscore(y_val, val_pred)
        df = pd.DataFrame({
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,

            'train_precision': train_precision,
            'val_precision': val_precision,
            'test_precision': test_precision,

            
            'train_recall': train_recall,
            'val_recall': val_recall,
            'test_recall': test_recall,

            
            'train_fscore': train_fscore,
            'val_fscore': val_fscore,
            'test_fscore': test_fscore,

            }, 
            index=['GaussianNB'])
        return df

    def qda_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_accuracy, train_precision, train_recall, train_fscore = self.accuracy_precision_recall_fscore(y_train, train_pred)
        test_accuracy, test_precision, test_recall, test_fscore = self.accuracy_precision_recall_fscore(y_test, test_pred)
        val_accuracy, val_precision, val_recall, val_fscore = self.accuracy_precision_recall_fscore(y_val, val_pred)
        df = pd.DataFrame({
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,

            'train_precision': train_precision,
            'val_precision': val_precision,
            'test_precision': test_precision,

            
            'train_recall': train_recall,
            'val_recall': val_recall,
            'test_recall': test_recall,
            
            'train_fscore': train_fscore,
            'val_fscore': val_fscore,
            'test_fscore': test_fscore,

            }, 
            index=['QuadraticDiscriminantAnalysis'])
        return df

    def lda_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = LinearDiscriminantAnalysis().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_accuracy, train_precision, train_recall, train_fscore = self.accuracy_precision_recall_fscore(y_train, train_pred)
        test_accuracy, test_precision, test_recall, test_fscore = self.accuracy_precision_recall_fscore(y_test, test_pred)
        val_accuracy, val_precision, val_recall, val_fscore = self.accuracy_precision_recall_fscore(y_val, val_pred)
        df = pd.DataFrame({
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,

            'train_precision': train_precision,
            'val_precision': val_precision,
            'test_precision': test_precision,

            
            'train_recall': train_recall,
            'val_recall': val_recall,
            'test_recall': test_recall,

            
            'train_fscore': train_fscore,
            'val_fscore': val_fscore,
            'test_fscore': test_fscore,
            }, 
            index=['LinearDiscriminantAnalysis'])
        return df
    

    def cat_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = CatBoostClassifier().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_accuracy, train_precision, train_recall, train_fscore = self.accuracy_precision_recall_fscore(y_train, train_pred)
        test_accuracy, test_precision, test_recall, test_fscore = self.accuracy_precision_recall_fscore(y_test, test_pred)
        val_accuracy, val_precision, val_recall, val_fscore = self.accuracy_precision_recall_fscore(y_val, val_pred)
        df = pd.DataFrame({
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,

            'train_precision': train_precision,
            'val_precision': val_precision,
            'test_precision': test_precision,
            
            'train_recall': train_recall,
            'val_recall': val_recall,
            'test_recall': test_recall,
            
            'train_fscore': train_fscore,
            'val_fscore': val_fscore,
            'test_fscore': test_fscore,
            }, 
            index=['CatBoostClassifier'])
        return df
    
    def clf_run(self):
        randomforest = self.rfc_train()
        gradientboosting = self.grad_train()
        xgboost = self.xgb_train()
        knn = self.knn_train()
        adaboost = self.ada_train()
        gaussian = self.gau_train()
        quadradictda = self.qda_train()
        lda = self.lda_train()
        catboost = self.cat_train()
        df = pd.concat([randomforest, gradientboosting, xgboost, knn, adaboost, gaussian, quadradictda, lda, catboost])
        df.index=['RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier', 'KNeighborsClassifier', 
                  'AdaBoostClassifier', 'GaussianNB', 'QuadraticDiscriminantAnalysis', 'LinearDiscriminantAnalysis', 'CatBoostClassifier']
        # return df.to_json(orient='records')
        print(df)