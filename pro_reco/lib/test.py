from ml_engine import Classification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd

df = pd.read_csv('F:\industry\data\clf_data\Test_01.csv')
df.drop(columns='STD_DT', axis=1, inplace=True)
df['INSP'] = LabelEncoder().fit_transform(df['INSP'])

# print(df)
X = df.drop(columns='INSP', axis=1)
y = df['INSP']

model = Classification(X, y).train_clf()
print(model['RandomForestClassifier'])
print(model['XGBClassifier'])
print(model['KNeighborsClassifier'])
print(model['AdaBoostClassifier'])
print(model['GradientBoostingClassifier'])
print(model['GaussianNB'])

