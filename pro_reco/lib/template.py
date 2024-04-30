import pandas as pd
import streamlit as st


class Template:
    def __init__(self, df):
        self.df = df

    def info_df(self):
        '''
        Columns name, Null, Dtype 확인
        '''
        df = self.df
        st.write('Null information')
        info_df = pd.DataFrame({'Column names': df.columns,
                                'Non-Null Count': df.count(),
                                'Null Count': df.isnull().sum(),
                                'Dtype': df.dtypes,
                                })
        info_df.reset_index(inplace=True)
        st.write(info_df.iloc[:, 1:].astype(str))

    def eda_df(self, tab_eda_df, tab_eda_info):
        '''
        DataFrame 확인을 위한 streamlit tab
        '''
        with tab_eda_df:
            st.write('Original data')
            st.dataframe(self.df)
        with tab_eda_info:
            # template = Template(df)
            self.info_df() 

    def label_to_drop(self, target_feture):
        '''
        제거할 Label 선택
        '''
        test = self.df[target_feture].value_counts().reset_index()
        val_counts_df = pd.DataFrame({'Labels': test.iloc[:, 0],
                                    'Counts': test.iloc[:, 1]})
        st.write(val_counts_df)
        bar_data = val_counts_df
        bar_data.index = val_counts_df['Labels']
        st.bar_chart(bar_data['Counts'])

        # Target Data 설정해야 제거할 Label 선택 가능
        label_to_drop = st.sidebar.multiselect('제거할 Target 데이터 선택', options=val_counts_df.iloc[:, 0])
        return label_to_drop
    
    def sample_df(self):
        '''
        Target column 설정하지 않았을 경우 보여주는 Sample data
        '''
        sample_df = pd.DataFrame({'Label': ['Select Label Column'], # Sample Data
                                'Counts': ['Select Label Column']})
        st.write(sample_df)
