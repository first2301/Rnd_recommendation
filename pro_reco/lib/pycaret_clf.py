# 함수형으로 실행하는 방법
# from pycaret.classification import *

# def train_caret_clf(data, target):
#     s = setup(data=data, target=target, session_id=123, normalize = True, normalize_method = 'minmax')
#     models = compare_models(sort='Accuracy')
#     result = pull()
#     prepro_compare_df = result.iloc[:, 2:].T # DataFrame transpose
#     prepro_compare_df.columns = result['Model']
#     return prepro_compare_df

# OOP 스타일로 실행하는 방법
import ray
from pycaret.classification import ClassificationExperiment

@ray.remote
def train_caret_clf(data, target):
    s = ClassificationExperiment()
    # s.setup(data=data, target=target, session_id=123, normalize = True, normalize_method = 'minmax')
    s.setup(data=data, target=target, session_id=123)
    
    models_to_exclude = ['lightgbm'] # 학습에 너무 오래 걸리는 모델
    models = s.compare_models(exclude=models_to_exclude, sort='F1') # exclude 옵션으로 일부 모델 제외, sort 옵션으로 정렬 기준으로 할 지표 정렬
    # models = s.compare_models(sort='F1') 

    # models 변수를 별도로 사용하지 않고 s.pull()로 머신러닝 모델 학습결과 출력 가능
    result = s.pull()
    prepro_compare_df = result.iloc[:, 1:].T 
    prepro_compare_df.columns = result['Model']
    return prepro_compare_df