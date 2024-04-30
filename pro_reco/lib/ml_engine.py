from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


class Classification:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_clf(self):
        X = self.X
        y = self.y

        rfc = RandomForestClassifier().fit(X, y)
        gb = GradientBoostingClassifier().fit(X, y)
        xgb = XGBClassifier().fit(X, y)
        knn = KNeighborsClassifier().fit(X, y)
        ada = AdaBoostClassifier().fit(X, y)
        gau = GaussianNB().fit(X, y)

        rfc_score = cross_val_score(rfc, X, y, cv=5).mean()
        xgb_score = cross_val_score(xgb, X, y, cv=5).mean()
        knn_score = cross_val_score(knn, X, y, cv=5).mean()
        ada_score = cross_val_score(ada, X, y, cv=5).mean()
        gb_score = cross_val_score(gb, X, y, cv=5).mean()
        gau_score = cross_val_score(gau, X, y, cv=5).mean()

        return {'RandomForestClassifier': rfc_score, 'XGBClassifier': xgb_score, 
                'KNeighborsClassifier': knn_score, 'AdaBoostClassifier': ada_score, 
                'GradientBoostingClassifier': gb_score, 'GaussianNB': gau_score}

# class ML:
#     def __init__(self, X_train, X_test, y_train, y_test):
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test

#     def train(self):
#         X_train = self.X_train
#         X_test = self.X_test
#         y_train = self.y_train
#         y_test = self.y_test
        
#         # rfc = RandomForestClassifier().fit(X_train, y_train)
#         # xgb = XGBClassifier().fit(X_train, y_train)
#         # knn = KNeighborsClassifier().fit(X_train, y_train)
#         # ada = AdaBoostClassifier().fit(X_train, y_train)

#         rfc = RandomForestClassifier()
#         xgb = XGBClassifier()
#         knn = KNeighborsClassifier()
#         ada = AdaBoostClassifier()

#         rfc_score = cross_val_score(X_train, y_train, cv=5)