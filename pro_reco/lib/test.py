from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd


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
class RegressionCurv:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def xgb(self):
        common_params = {
            "X": X,
            "y": y,
            "train_sizes": np.linspace(0.1, 1.0, 5),
            "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
            "n_jobs": 4,
            "return_times": True,
        }
        
        xgb = XGBRegressor().fit()

        train_sizes, _, test_scores_nb, fit_times_nb, score_times_nb = learning_curve(
            xgb, **common_params
        )
        

PATH = 'F:\industry\data\clf_data\Test_02.csv'

df = pd.read_csv(PATH)
X = df.drop('Potability', axis=1)
y = df['Potability']

# result = RegressionCurv(X, y).train_reg()
# print(result)

common_params = {
    "X": X,
    "y": y,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "n_jobs": 4,
    "return_times": True,
}

xgb = XGBRegressor()

train_sizes, _, test_scores_nb, fit_times_nb, score_times_nb = learning_curve(
    xgb, **common_params
)

print(train_sizes, _, test_scores_nb, fit_times_nb, score_times_nb)