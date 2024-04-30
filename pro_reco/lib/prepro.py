from sklearn.preprocessing import LabelEncoder

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
    
