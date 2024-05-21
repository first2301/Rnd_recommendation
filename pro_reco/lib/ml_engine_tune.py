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


class Classification:
    def __init__(self, df, target):
        self.df = df
        self.target = target

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
        return df.to_json(orient='records')
    
class Regression: 
    def __init__(self, df, target):
        self.df = df
        self.target = target

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

    def mae_mse_errors(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return mae, mse

    def xgb_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = XGBRegressor().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_mae, train_mse = self.mae_mse_errors(y_train, train_pred)
        test_mae, test_mse = self.mae_mse_errors(y_test, test_pred)
        val_mae, val_mse = self.mae_mse_errors(y_val, val_pred)

        df = pd.DataFrame({
            'train_accuracy': train_mae,
            'test_accuracy': test_mae,
            'val_accuracy': val_mae,
            
            'train_mse': train_mse,
            'test_mse': test_mse,
            'val_mse': val_mse,
            }, 
            index=['XGBRegressor'])
        
        return df
    
    def rfc_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = RandomForestRegressor().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_mae, train_mse = self.mae_mse_errors(y_train, train_pred)
        test_mae, test_mse = self.mae_mse_errors(y_test, test_pred)
        val_mae, val_mse = self.mae_mse_errors(y_val, val_pred)

        df = pd.DataFrame({
            'train_accuracy': train_mae,
            'test_accuracy': test_mae,
            'val_accuracy': val_mae,
            
            'train_mse': train_mse,
            'test_mse': test_mse,
            'val_mse': val_mse,
            }, 
            index=['RandomForestRegressor'])
        
        return df
    
    def ada_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = AdaBoostRegressor().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_mae, train_mse = self.mae_mse_errors(y_train, train_pred)
        test_mae, test_mse = self.mae_mse_errors(y_test, test_pred)
        val_mae, val_mse = self.mae_mse_errors(y_val, val_pred)

        df = pd.DataFrame({
            'train_accuracy': train_mae,
            'test_accuracy': test_mae,
            'val_accuracy': val_mae,
            
            'train_mse': train_mse,
            'test_mse': test_mse,
            'val_mse': val_mse,
            }, 
            index=['AdaBoostRegressor'])
        
        return df
    
    def grad_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = GradientBoostingRegressor().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_mae, train_mse = self.mae_mse_errors(y_train, train_pred)
        test_mae, test_mse = self.mae_mse_errors(y_test, test_pred)
        val_mae, val_mse = self.mae_mse_errors(y_val, val_pred)

        df = pd.DataFrame({
            'train_accuracy': train_mae,
            'test_accuracy': test_mae,
            'val_accuracy': val_mae,
            
            'train_mse': train_mse,
            'test_mse': test_mse,
            'val_mse': val_mse,
            }, 
            index=['GradientBoostingRegressor'])
        
        return df
    
    def ridge_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = Ridge().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_mae, train_mse = self.mae_mse_errors(y_train, train_pred)
        test_mae, test_mse = self.mae_mse_errors(y_test, test_pred)
        val_mae, val_mse = self.mae_mse_errors(y_val, val_pred)

        df = pd.DataFrame({
            'train_accuracy': train_mae,
            'test_accuracy': test_mae,
            'val_accuracy': val_mae,
            
            'train_mse': train_mse,
            'test_mse': test_mse,
            'val_mse': val_mse,
            }, 
            index=['Ridge'])
        
        return df
    
    def lasso_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = Lasso().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_mae, train_mse = self.mae_mse_errors(y_train, train_pred)
        test_mae, test_mse = self.mae_mse_errors(y_test, test_pred)
        val_mae, val_mse = self.mae_mse_errors(y_val, val_pred)

        df = pd.DataFrame({
            'train_accuracy': train_mae,
            'test_accuracy': test_mae,
            'val_accuracy': val_mae,
            
            'train_mse': train_mse,
            'test_mse': test_mse,
            'val_mse': val_mse,
            }, 
            index=['Lasso'])
        
        return df

    def elnet_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = ElasticNet().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_mae, train_mse = self.mae_mse_errors(y_train, train_pred)
        test_mae, test_mse = self.mae_mse_errors(y_test, test_pred)
        val_mae, val_mse = self.mae_mse_errors(y_val, val_pred)

        df = pd.DataFrame({
            'train_accuracy': train_mae,
            'test_accuracy': test_mae,
            'val_accuracy': val_mae,
            
            'train_mse': train_mse,
            'test_mse': test_mse,
            'val_mse': val_mse,
            }, 
            index=['ElasticNet'])
        
        return df
    
    def knn_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = KNeighborsRegressor().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_mae, train_mse = self.mae_mse_errors(y_train, train_pred)
        test_mae, test_mse = self.mae_mse_errors(y_test, test_pred)
        val_mae, val_mse = self.mae_mse_errors(y_val, val_pred)

        df = pd.DataFrame({
            'train_accuracy': train_mae,
            'test_accuracy': test_mae,
            'val_accuracy': val_mae,
            
            'train_mse': train_mse,
            'test_mse': test_mse,
            'val_mse': val_mse,
            }, 
            index=['KNeighborsRegressor'])
        
        return df
    
    def cat_train(self):
        X_train, X_test, y_train, y_test = self.train_test_set()
        X_train, X_val, y_train, y_val = self.train_val_set()
        model = CatBoostRegressor().fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        train_mae, train_mse = self.mae_mse_errors(y_train, train_pred)
        test_mae, test_mse = self.mae_mse_errors(y_test, test_pred)
        val_mae, val_mse = self.mae_mse_errors(y_val, val_pred)

        df = pd.DataFrame({
            'train_accuracy': train_mae,
            'test_accuracy': test_mae,
            'val_accuracy': val_mae,
            
            'train_mse': train_mse,
            'test_mse': test_mse,
            'val_mse': val_mse,
            }, 
            index=['CatBoostRegressor'])
        
        return df

    def reg_run(self):
        xgboost = self.xgb_train()
        randomforest = self.rfc_train()
        adaboost = self.ada_train()
        gradientboost = self.grad_train()
        ridge = self.ridge_train()
        lassso = self.lasso_train()
        elasticnet = self.elnet_train()
        knn = self.knn_train()
        catboost = self.cat_train()
        df = pd.concat([xgboost, randomforest, adaboost, gradientboost, ridge, lassso, elasticnet, knn, catboost])
        df.index=['XGBRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'GradientBoostingRegressor', 'Ridge', 'Lasso', 'ElasticNet', 'KNeighborsRegressor', 'CatBoostRegressor']
        return df.to_json(orient='records')


class Cluster:
    def __init__(self, data):
        self.data = data

    def kmeans_train(self):
        data = self.data
        model = KMeans()
        score = silhouette_score(data, model.fit_predict(data))
        return score
    
    def dbscan_train(self):
        data = self.data
        model = DBSCAN(eps=0.5, min_samples=5)
        score = silhouette_score(data, model.fit_predict(data))
        return score
    
    def affinity_train(self):
        data = self.data
        model = AffinityPropagation()
        score = silhouette_score(data, model.fit_predict(data))
        return score
    
    def meanshift_train(self):
        data = self.data
        model = MeanShift()
        score = silhouette_score(data, model.fit_predict(data))
        return score
    
    def spec_train(self):
        data = self.data
        model = SpectralClustering()
        score = silhouette_score(data, model.fit_predict(data))
        return score
    
    def specbi_train(self):
        data = self.data
        model = SpectralBiclustering()
        score = silhouette_score(data, model.fit_predict(data))
        return score
    
    def birch_train(self):
        data = self.data
        model = Birch()
        score = silhouette_score(data, model.fit_predict(data))
        return score
    
    def agg_train(self):
        data = self.data
        model = AgglomerativeClustering()
        score = silhouette_score(data, model.fit_predict(data))
        return score
    
    def clu_run(self):
        kmeans = self.kmeans_train()
        dbscan = self.dbscan_train()
        affinitypropagation = self.affinity_train()
        meanshift = self.meanshift_train()
        spectralclustering = self.spec_train()
        spectralBiclustering = self.specbi_train()
        birch = self.birch_train()
        agglomerativeclustering = self.agg_train()

        df = pd.concat([kmeans, dbscan, affinitypropagation, meanshift, spectralclustering,
                        spectralBiclustering, birch, agglomerativeclustering])
        df.index=['KMeans', 'DBSCAN', 'AffinityPropagation', 'MeanShift', 'spectralclustering',
                  'SpectralBiclustering', 'Birch', 'AgglomerativeClustering']
        
        return df.to_json(orient='records')
    

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

#     def train_reg(self):
#         xgb_result = self.xgb()

#         return xgb_result
