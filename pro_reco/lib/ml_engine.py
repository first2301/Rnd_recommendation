# ensemble
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.cluster import KMeans, DBSCAN
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_fscore_support
import pandas as pd

# class Classification:
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def train_clf(self):
#         X = self.X
#         y = self.y

#         rfc = RandomForestClassifier().fit(X, y)
#         gb = GradientBoostingClassifier().fit(X, y)
#         xgb = XGBClassifier().fit(X, y)
#         knn = KNeighborsClassifier().fit(X, y)
#         ada = AdaBoostClassifier().fit(X, y)
#         gau = GaussianNB().fit(X, y)

#         rfc_score = cross_val_score(rfc, X, y, cv=5).mean()
#         xgb_score = cross_val_score(xgb, X, y, cv=5).mean()
#         knn_score = cross_val_score(knn, X, y, cv=5).mean()
#         ada_score = cross_val_score(ada, X, y, cv=5).mean()
#         gb_score = cross_val_score(gb, X, y, cv=5).mean()
#         gau_score = cross_val_score(gau, X, y, cv=5).mean()

#         return {'RandomForestClassifier': rfc_score, 'XGBClassifier': xgb_score, 
#                 'KNeighborsClassifier': knn_score, 'AdaBoostClassifier': ada_score, 
#                 'GradientBoostingClassifier': gb_score, 'GaussianNB': gau_score}

class Classification:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_clf(self):
        X = self.X
        y = self.y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rfc = RandomForestClassifier().fit(X_train, y_train)
        gb = GradientBoostingClassifier().fit(X_train, y_train)
        xgb = XGBClassifier().fit(X_train, y_train)
        knn = KNeighborsClassifier().fit(X_train, y_train)
        ada = AdaBoostClassifier().fit(X_train, y_train)
        gau = GaussianNB().fit(X_train, y_train)
        qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train) 
        lda= LinearDiscriminantAnalysis().fit(X_train, y_train)
        cat = CatBoostClassifier().fit(X_train, y_train)
        

        
        rfc_pred = rfc.predict(X_test)
        gb_pred = gb.predict(X_test)
        xgb_pred = xgb.predict(X_test)
        knn_pred = knn.predict(X_test)
        ada_pred = ada.predict(X_test)
        gau_pred = gau.predict(X_test)
        qda_pred = qda.predict(X_test)
        lda_pred = lda.predict(X_test)
        cat_pred = cat.predict(X_test)

        rfc_score = precision_recall_fscore_support(y_test, rfc_pred, average='micro')
        gb_score = precision_recall_fscore_support(y_test, gb_pred, average='micro')
        xgb_score = precision_recall_fscore_support(y_test, xgb_pred, average='micro')
        knn_score = precision_recall_fscore_support(y_test, knn_pred, average='micro')
        ada_score = precision_recall_fscore_support(y_test, ada_pred, average='micro')
        gau_score = precision_recall_fscore_support(y_test, gau_pred, average='micro')
        qda_score = precision_recall_fscore_support(y_test, gau_pred, average='micro')
        lda_score = precision_recall_fscore_support(y_test, gau_pred, average='micro')
        cat_score = precision_recall_fscore_support(y_test, gau_pred, average='micro')

        rfc_acc, gb_acc, xgb_acc = accuracy_score(y_test, rfc_pred), accuracy_score(y_test, gb_pred), accuracy_score(y_test, xgb_pred) 
        knn_acc, ada_acc, gau_acc = accuracy_score(y_test, knn_pred), accuracy_score(y_test, ada_pred), accuracy_score(y_test, gau_pred)
        qda_acc, lda_acc, cat_acc = accuracy_score(y_test, qda_pred), accuracy_score(y_test, lda_pred), accuracy_score(y_test, cat_pred)

        df = pd.DataFrame({'accuracy': [rfc_acc, xgb_acc, knn_acc, ada_acc, gb_acc, gau_acc, qda_acc, lda_acc, cat_acc],
                           'precision': [rfc_score[0], xgb_score[0], knn_score[0], ada_score[0], gb_score[0], gau_score[0], qda_score[0], lda_score[0], cat_score[0]], 
                           'recall': [rfc_score[1], xgb_score[1], knn_score[1], ada_score[1], gb_score[1], gau_score[1], qda_score[1], lda_score[1], cat_score[1]], 
                           'f1_score': [rfc_score[2], xgb_score[2], knn_score[2], ada_score[2], gb_score[2], gau_score[2], qda_score[2], lda_score[2], cat_score[2]]}, 
                           index=['RandomForestClassifier', 'XGBClassifier', 'KNeighborsClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'GaussianNB',
                                  'QuadraticDiscriminantAnalysis', 'LinearDiscriminantAnalysis', 'CatBoostClassifier'])
        
        return df.to_json(orient='records')

class Cluster:
    def __init__(self, data):
        self.data = data

    def train_clu(self):
        data = self.data
        km = KMeans()
        db = DBSCAN()

        km_score = silhouette_score(data, km.fit_predict(data))
        db_score = silhouette_score(data, db.fit_predict(data))
        return {'KMeans': km_score, 'DBSCAN': db_score}

class Regression:
    def __init__(self, X, y):
        self.X = X
        self.y = y    
    def train_reg(self):
        X = self.X
        y = self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        xgb = XGBRegressor().fit(X_train, y_train)
        rfc = RandomForestRegressor().fit(X_train, y_train)
        ada = AdaBoostRegressor().fit(X_train, y_train)
        gb = GradientBoostingRegressor().fit(X_train, y_train)
        rg = Ridge().fit(X_train, y_train)
        ls = Lasso().fit(X_train, y_train)
        en = ElasticNet().fit(X_train, y_train)
        knn = KNeighborsRegressor().fit(X_train, y_train)
        cat = CatBoostRegressor().fit(X_train, y_train)

        xgb_pred = xgb.predict(X_test)
        rfc_pred = rfc.predict(X_test)
        ada_pred = ada.predict(X_test)
        gb_pred = gb.predict(X_test)
        rg_pred = rg.predict(X_test)
        ls_pred = ls.predict(X_test)
        en_pred = en.predict(X_test)
        knn_pred = knn.predict(X_test)
        cat_pred = cat.predict(X_test)

        xgb_mae, rfc_mae, ada_mae, gb_mae = mean_absolute_error(y_test, xgb_pred), mean_absolute_error(y_test, rfc_pred), mean_absolute_error(y_test, ada_pred), mean_absolute_error(y_test, gb_pred)
        rg_mae, ls_mae, en_mae = mean_absolute_error(y_test, rg_pred), mean_absolute_error(y_test, ls_pred), mean_absolute_error(y_test, en_pred)
        knn_mae, cat_mae = mean_absolute_error(y_test, knn_pred), mean_absolute_error(y_test, cat_pred)
       
        xgb_mse, rfc_mse, ada_mse, gb_mse = mean_squared_error(y_test, xgb_pred), mean_squared_error(y_test, rfc_pred), mean_squared_error(y_test, ada_pred), mean_squared_error(y_test, gb_pred)
        rg_mse, ls_mse, en_mse = mean_squared_error(y_test, rg_pred), mean_squared_error(y_test, ls_pred), mean_squared_error(y_test, en_pred)
        knn_mse, cat_mse = mean_squared_error(y_test, knn_pred), mean_squared_error(y_test, cat_pred)

        df = pd.DataFrame({'mean_absolute_error': [xgb_mae, rfc_mae, ada_mae, gb_mae, rg_mae, ls_mae, en_mae, knn_mae, cat_mae],
                           'mean_squared_error': [xgb_mse, rfc_mse, ada_mse, gb_mse, rg_mse, ls_mse, en_mse, knn_mse, cat_mse]},
                           index=['XGBRegressor', 'RandomForestRegressor', 'AdaBoostRegressor',
                                  'GradientBoostingRegressor', 'Ridge', 'Lasso', 'ElasticNet', 'KNeighborsRegressor', 'CatBoostRegressor'])


        return df.to_json(orient='records')
