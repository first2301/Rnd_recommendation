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