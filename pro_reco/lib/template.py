import pandas as pd
import streamlit as st
import plotly.express as px

class Template:
    def __init__(self, df):
        self.df = df

    def info_df(self):
        '''
        Columns name, Null, Dtype 확인
        '''
        df = self.df
        # st.write('Null information')
        info_df = pd.DataFrame({'Column names': df.columns,
                                'Non-Null Count': df.count(),
                                'Null Count': df.isnull().sum(),
                                'Dtype': df.dtypes,
                                })
        info_df.reset_index(inplace=True)
        st.dataframe(info_df.iloc[:, 1:].astype(str), use_container_width=True)

    def eda_df(self, tab_eda_df, tab_eda_info):
        '''
        DataFrame 확인을 위한 streamlit tab
        '''
        with tab_eda_df:
            # st.write('Original data')
            st.dataframe(self.df, use_container_width=True)
        with tab_eda_info:
            # template = Template(df)
            self.info_df() 

    def label_to_drop(self, target_feture):
        '''
        제거할 Label 선택
        '''
        val_counts = self.df[target_feture].value_counts().reset_index()
        val_counts_df = pd.DataFrame({'Labels': val_counts.iloc[:, 0],
                                    'Counts': val_counts.iloc[:, 1]})
        bar_data = val_counts_df
        bar_data.index = val_counts_df['Labels']
        # st.dataframe(val_counts_df, use_container_width=True) # Target Data Counts
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(bar_data['Counts'])
            st.dataframe(val_counts_df, use_container_width=True) # Target Data Counts
        with col2:
            pie_fig = px.pie(bar_data, values=bar_data['Counts'], names=bar_data['Labels'])
            st.plotly_chart(pie_fig)
        # st.dataframe(val_counts_df, use_container_width=True) # Target Data Counts
            
        
        # Target Data 설정해야 제거할 Label 선택 가능
        label_to_drop = st.sidebar.multiselect('제거할 Target 데이터 선택', options=val_counts_df.iloc[:, 0])
        return label_to_drop
    
    def sample_df(self):
        '''
        Target column 설정하지 않았을 경우 보여주는 Sample data
        '''
        sample_df = pd.DataFrame({'Label': ['Select Label Column'], # Sample Data
                                'Counts': ['Select Label Column']})
        st.dataframe(sample_df, use_container_width=True)

    def print_best_result(
            self, title_1, title_2, title_3, title_4, 
            best_score_df_1, best_score_df_2, best_score_df_3, best_score_df_4
            ):
        '''
        Insert title_1, title_2, best_score_df_1, best_score_df_2
        '''
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(title_1)
                in_col1, in_col2 = st.columns(2)
                
                with in_col1:
                    st.bar_chart(best_score_df_1)
                with in_col2:
                    st.dataframe(best_score_df_1, use_container_width=True)

            with col2:
                st.subheader(title_2)
                in_col1, in_col2 = st.columns(2)
                with in_col1:
                    st.bar_chart(best_score_df_2)
                with in_col2:
                    st.dataframe(best_score_df_2, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(title_3)
                in_col1, in_col2 = st.columns(2)
                
                with in_col1:
                    st.bar_chart(best_score_df_3)
                with in_col2:
                    st.dataframe(best_score_df_3, use_container_width=True)

            with col2:
                st.subheader(title_4)
                in_col1, in_col2 = st.columns(2)
                with in_col1:
                    st.bar_chart(best_score_df_4)
                with in_col2:
                    st.dataframe(best_score_df_4, use_container_width=True)

    def print_reg_best_result(
                self, title_1, title_2, 
                best_score_df_1, best_score_df_2,
                ):
        '''
        Insert title_1, title_2, best_score_df_1, best_score_df_2
        '''
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(title_1)
                in_col1, in_col2 = st.columns(2)
                
                with in_col1:
                    st.bar_chart(best_score_df_1)
                with in_col2:
                    st.dataframe(best_score_df_1, use_container_width=True)

            with col2:
                st.subheader(title_2)
                in_col1, in_col2 = st.columns(2)
                with in_col1:
                    st.bar_chart(best_score_df_2)
                with in_col2:
                    st.dataframe(best_score_df_2, use_container_width=True)



        # with st.container():
        #     col1, col2 = st.columns(2)
        #     with col1:
        #         st.subheader(title_1)
        #         in_col1, in_col2 = st.columns(2)
                
        #         with in_col1:
        #             st.bar_chart(best_score_df_1)
        #         with in_col2:
        #             st.write(best_score_df_1)
        #     with col2:
        #         st.subheader(title_2)
        #         in_col1, in_col2 = st.columns(2)
        #         with in_col1:
        #             st.bar_chart(best_score_df_2)
        #         with in_col2:
        #             st.write(best_score_df_2)
                    
        # with st.container():
        #     col1, col2 = st.columns(2)
        #     with col2:
        #         st.subheader(title_3)
        #         in_col1, in_col2 = st.columns(2)
        #         with in_col1:
        #             st.bar_chart(best_score_df_3)
        #         with in_col2:
        #             st.write(best_score_df_3)
        #     with col2:
        #         st.subheader(title_4)
        #         in_col1, in_col2 = st.columns(2)
        #         with in_col1:
        #             st.bar_chart(best_score_df_4)
        #         with in_col2:
        #             st.write(best_score_df_4)

    def print_trial_result(
            self, title_1, title_2, title_3, title_4,
            trial_score_df_1, trial_score_df_2, trial_score_df_3, trial_score_df_4
            ):
        '''
        Insert title_1, title_2, trial_score_df_1, trial_score_df_2
        '''

        with st.container():
            st.subheader(title_1)
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(trial_score_df_1)
            with col2:
                st.write(trial_score_df_1)

        with st.container():
            st.subheader(title_2)
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(trial_score_df_2)
            with col2:
                st.write(trial_score_df_2)

        with st.container():
            st.subheader(title_3)
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(trial_score_df_3)
            with col2:
                st.write(trial_score_df_3)

        with st.container():
            st.subheader(title_4)
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(trial_score_df_4)
            with col2:
                st.write(trial_score_df_4)

    def print_reg_trial_result(
        self, title_1, title_2,
        trial_score_df_1, trial_score_df_2, 
        ):
        '''
        Insert title_1, title_2, trial_score_df_1, trial_score_df_2
        '''

        with st.container():
            st.subheader(title_1)
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(trial_score_df_1)
            with col2:
                st.write(trial_score_df_1)

        with st.container():
            st.subheader(title_2)
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(trial_score_df_2)
            with col2:
                st.write(trial_score_df_2)

    # def print_trial_result(self, title_1, title_2, trial_score_df_1, trial_score_df_2):
    #     '''
    #     Insert title_1, title_2, trial_score_df_1, trial_score_df_2
    #     '''
    #     with st.container():
    #     # st.subheader('mean_squared_error')
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             st.subheader(title_1)
    #             in_col1, in_col2 = st.columns(2)
                
    #             with in_col1:
    #                 st.line_chart(trial_score_df_1)
    #             with in_col2:
    #                 st.write(trial_score_df_1)
    #         with col2:
    #             st.subheader(title_2)
    #             in_col1, in_col2 = st.columns(2)
    #             with in_col1:
    #                 st.line_chart(trial_score_df_2)
    #             with in_col2:
    #                 st.write(trial_score_df_2)