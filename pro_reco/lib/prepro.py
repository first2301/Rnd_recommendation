from sklearn.preprocessing import LabelEncoder
import pandas as pd

class Preprocessing:
    def __init__(self):
        self.le = LabelEncoder()

    def label_encoding(self, data):
        '''
        data에 labelencoding 실행할 column 전체 할당
        '''
        le = self.le
        le.fit(data) 
        return le.transform(data)        

    def encoded_df(self, df, label):
        '''
        df, label 입력
        '''
        df = df
        df[label] = self.label_encoding(df[label])
        return df
    
    def is_best_trial_df(self, best_trial):
        best_trial_df = pd.DataFrame([{
                            'number': best_trial.number,
                            'value': best_trial.value,
                            # 'state': t.state,
                            'params': best_trial.params,
                            'datetime_start': best_trial.datetime_start,
                            'datetime_complete': best_trial.datetime_complete,
                        }])
        best_params_df = best_trial_df['params'].apply(pd.Series)
        best_trial_df = pd.concat([best_trial_df.drop(columns=['params']), best_params_df], axis=1)
        return best_trial_df
    
    def is_all_trials_df(self, trials):
        trials_df = pd.DataFrame([{
                            'number': t.number,
                            'value': t.value,
                            # 'state': t.state,
                            'params': t.params,
                            'datetime_start': t.datetime_start,
                            'datetime_complete': t.datetime_complete,
                        } for t in trials])
        trials_params_df = trials_df['params'].apply(pd.Series)
        trials_df = pd.concat([trials_df.drop(columns=['params']), trials_params_df], axis=1)
        return trials_df
        
    def make_dict(self, result):
        trial_df = dict()
        best_df = dict()
        model_list = ['randomforest', 'gradient', 'xgboost', 'catboost', 'adaboost', 'knn', 'gaussian']
        
        for n in model_list:
            best_df[n] = self.is_best_trial_df(result[n]['best_trial']).to_json()
            trial_df[n] = self.is_all_trials_df(result[n]['trials']).to_json()

        # for n in range(0, len(result)):
        #     best_df[n] = self.is_best_trial_df(result[n]['best_trial']).to_json()
        #     trial_df[n] = self.is_all_trials_df(result[n]['trials']).to_json()

        return {'best_df': best_df, 'trial_df': trial_df}
    
    def make_dataframe(self, result):
        best_df = pd.DataFrame()
        model_list = ['randomforest', 'gradient', 'xgboost', 'catboost', 'adaboost', 'knn', 'gaussian']
        
        for n in model_list:
            temp_df = self.is_best_trial_df(result[n]['best_trial'])
            best_df = pd.concat([best_df, ])

        return best_df