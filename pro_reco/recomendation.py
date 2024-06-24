# 대시보드 UI
# https://docs.streamlit.io/knowledge-base/deploy/increase-file-uploader-limit-streamlit-cloud
# https://docs.streamlit.io/library/advanced-features/configuration#set-configuration-options
# 실행방법1: streamlit run recomendation.py
# 실행방법2: streamlit run recomendation.py --server.maxUploadSize 500 --server.maxMessageSize 500 (업로드 파일 용량 증대할 경우)
# import time # 코드 실행 시간 측정 시 사용
# sqlite:///./database/database.db
# AxiosError: Request failed with status code 403 in streamlit 발생 시, enableXsrfProtection 입력하여 실행
# streamlit run recomendation.py --server.enableXsrfProtection false
import ray
import json
import requests
import pandas as pd
import streamlit as st
from lib.template import Template
from lib.prepro import Preprocessing
from database.connector import Database # , SelectTB
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(layout="wide")
st.sidebar.title("Details")

# 분류, 이상 탐지 등 추천받을 머신러닝 모델 선택
option = st.sidebar.selectbox(
    'Select Machine Learning Task', ('분류', '이상탐지', '회귀'))
connecton_option = st.sidebar.selectbox(
    'Select how to upload data', ('File_upload', 'DB_connection'))
uploaded_file = None
df = None
if connecton_option == 'File_upload':
    uploaded_file = st.sidebar.file_uploader("csv file upload", type="csv") # 파일 업로드
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
if connecton_option == 'DB_connection':
    database_string = st.sidebar.text_input('Input database string') # Database string 입력
    try:
        db = Database(database_string)
        tb_info = db.tables_info() 
        table_names = [table_names for table_names in tb_info]
        table_meta_df = pd.DataFrame({'table_names': table_names})
        st.subheader('테이블 목록')
        st.write(table_meta_df)
    except:
        st.sidebar.write('Need to input Database string')
    table_name = st.sidebar.text_input('Input table name') # 데이터 전처리 수행할 테이블명 입력
    # select_tb = SelectTB(database_string, table_name).select_tb()
    try:
        df = db.db_to_df(table_name)
    except:
        st.sidebar.write('Need to input table name')

# @st.cache_data # resource
# def load_data(uploaded_file):
#     return pd.read_csv(uploaded_file)
# try:
#     if connecton_option == 'File_upload':
#         df = load_data(uploaded_file)  
# except ValueError:
#     st.sidebar.write('Need to upload csv file')

with st.spinner('Wait for it...'):
    # if uploaded_file is None: 
    #     st.write(
    #     '''
    #     ### 머신러닝 실행 방법
    #     * 분류
    #     1. Upload csv file 
    #     2. Select Target column 
    #     3. Drop cloumns
    #     4. 제거할 Target 데이터 선택

    #     * 이상 탐지
    #     1. Upload csv file
    #     2. 머신러닝 테스트 실행
    #     ''')

    updated_df = None
    # Uploaded data Dashboard
    if uploaded_file is not None or df is not None:
        template = Template(df)
        st.subheader('데이터 분석')
        col_list = df.columns.tolist() # 데이터 전처리 옵션 설정 리스트
        target_feture = ""
        if option == '분류' or option == '회귀':
            target_feture = st.sidebar.multiselect('Select Target Column', options=col_list)
        data_to_drop = st.sidebar.multiselect('Drop Cloumns', options=col_list)
        data_for_labelencoding = st.sidebar.multiselect('Choose LabelEncoding column name', options=col_list)
        
        tab_eda_df, tab_eda_info, tab_Label_counts = st.tabs(['Original data', 'Null information', 'Target Data Counts']) # tab_Label_counts Labels counts
        # tab_eda_df, tab_eda_info tab UI Template
        template.eda_df(tab_eda_df=tab_eda_df, tab_eda_info=tab_eda_info)
        label_to_drop = ""
        with tab_Label_counts: # Target Data 정보 출력 및 시각화
            if target_feture:            
                label_to_drop = template.label_to_drop(target_feture) # 제거할 Target 데이터 선택
            else:
                template.sample_df()

        # LaberEncooding
        if data_for_labelencoding:
            prepro = Preprocessing()
            if updated_df is None:
                # st.write(type(df[data_for_labelencoding]))
                df = prepro.encoded_df(df, data_for_labelencoding[0])
                updated_df = df
            if updated_df is not None:
                updated_df = prepro.encoded_df(updated_df, data_for_labelencoding[0])

        # 선택한 Column 제거   
        if data_to_drop:
            for data in data_to_drop:
                updated_df = df.drop(data_to_drop, axis=1)

        # 선택한 Target Data 제거
        try:
            if label_to_drop:
                target_feture = target_feture[0]
                label_to_drop = label_to_drop[0]
                updated_df = df[df[target_feture] != label_to_drop]
        except ValueError:
            st.write('1개 이상 데이터가 남아있어야 합니다.')

        # 데이터 전처리된 데이터 출력
        if updated_df is not None: 
            st.subheader('데이터 전처리')
            st.dataframe(updated_df)
        
        if st.sidebar.button("초기화"):
            st.cache_resource.clear()


#################### Starting ML traning
        button_for_training = st.sidebar.button("머신러닝 테스트 실행", key="button1") 
        if button_for_training: # 분류, 이상탐지 옵션에 따라 머신러닝 학습 진행
            # start_time = time.time() # 학습 시간 체크 시 설정

            if option == '분류':
                st.subheader('머신러닝 학습 결과')
                with st.spinner('Wait for it...'):
                    if updated_df is None:
                        updated_df = df   

                    json_data = updated_df.to_json() # pandas DataFrame를 json 형태로 변환
                    data_dump = json.dumps({'json_data':json_data, 'target': target_feture}) # 학습 데이터, Target Data 객체를 문자열로 직렬화(serialize)
                    data = json.loads(data_dump) # json을 파이썬 객체로 변환

                    # response = requests.post('http://127.0.0.1:8001/clf', json=data)
                    response = requests.post('http://127.0.0.1:8001/new_clf', json=data) 
                    if response.status_code == 200: 
                        json_data = response.json() 
                        # model_compare_clf = json_data['result']
                        data = json.loads(json_data['result'])
                        
                        # st.write('data',data)
                        model_list = ['randomforest', 'gradient', 'xgboost', 'catboost', 'adaboost', 'knn', 'gaussian']
                        
                        # temp_df = pd.DataFrame([], columns=['accuracy'],index=['randomforest', 'gradient', 'xgboost', 'catboost', 'adaboost', 'knn', 'gaussian'])
                        temp_best_df = pd.DataFrame()
                        temp_trial_df = pd.DataFrame()
                        # best_df = pd.DataFrame()
                        for name in model_list:
                            loaded_best_data = json.loads(data['best_df'][name])
                            loaded_trial_data = json.loads(data['trial_df'][name])
                            # best score
                            loaded_best_df = pd.DataFrame(loaded_best_data)
                            loaded_best_series = loaded_best_df['value']
                            temp_best_df = pd.concat([temp_best_df, loaded_best_series])
                            # trial score

                            # loaded_trial_df = pd.DataFrame(loaded_trial_data)
                            # loaded_trial_series = loaded_trial_df['value']
                            # temp_trial_df = pd.concat([temp_trial_df, loaded_trial_series], axis=1)

                            loaded_trial_df = pd.DataFrame(loaded_trial_data)
                            loaded_trial_series = loaded_trial_df['value']
                            temp_trial_df[name + '_accuracy'] = loaded_trial_series

                        temp_best_df.columns = ['best_accuracy']
                        temp_best_df.index = model_list

                        with st.container():
                            st.subheader('accuracy')
                            col1, col2 = st.columns(2)
                            with col1:
                                tab1, tab2 = st.tabs(['Best_accuracy_bar_chart', 'Trial_accuracy_line_chart'])
                                with tab1:
                                    st.bar_chart(temp_best_df)
                                with tab2:
                                    st.line_chart(temp_trial_df)

                            with col2:
                                tab1, tab2 = st.tabs(['trial_accuracy', 'best_accuracy'])
                                with tab1:
                                    num_rows_to_display = 7  # 원하는 행의 수
                                    row_height = 50  # 각 행의 높이
                                    total_height = num_rows_to_display * row_height
                                    st.dataframe(temp_trial_df,  height=total_height)
                                with tab2:
                                    st.write(temp_best_df)

                        # col1, col2 = st.columns(2)
                        # tab1, tab2 = st.tabs(['trial_accuracy', 'best_accuracy'])
                        # with tab1:
                        #     st.write(temp_trial_df)
                        # with tab2:
                        #     st.write(temp_best_df)
                        
                        # temp_trial_df.columns = ['accuracy']
                        # temp_trial_df.index = model_list


                        # st.write(temp_trial_df)          
                        
            if option == '회귀':
                st.subheader('머신러닝 학습 결과')
                with st.spinner('Wait for it...'):
                    if updated_df is None:
                        updated_df = df   

                    json_data = updated_df.to_json() # pandas DataFrame를 json 형태로 변환
                    data_dump = json.dumps({'json_data':json_data, 'target': target_feture}) # 학습 데이터, Target Data 객체를 문자열로 직렬화(serialize)
                    data = json.loads(data_dump) # json을 파이썬 객체로 변환

                    response = requests.post('http://127.0.0.1:8001/new_reg', json=data) 
                     
                    if response.status_code == 200: 
                        json_data = response.json() # NIPA 서버에서 학습한 데이터를 json으로 response 
                        # model_compare_clf = json_data['result']
                        data = json.loads(json_data['result'])
                        # st.write(data)
                        st.write(data)
                    else:
                        st.write("Error:", response.status_code)

                        # model_compare_clf = pd.DataFrame(data, index=['XGBRegressor', 'RandomForestRegressor', 'AdaBoostRegressor',
                        #                                               'GradientBoostingRegressor', 'Ridge', 'Lasso', 'ElasticNet'])


                        # with st.container():
                        #     col1, col2 = st.columns(2)
                        #     with col1:
                        #         st.subheader('mean_absolute_error')
                        #         st.bar_chart(model_compare_clf['mean_absolute_error'])
                        #     with col2:
                        #         st.subheader('mean_absolute_error')
                        #         st.dataframe(model_compare_clf['mean_absolute_error'])

                        # with st.container():
                        #     col1, col2 = st.columns(2)
                        #     with col1:
                        #         st.subheader('mean_squared_error')
                        #         st.bar_chart(model_compare_clf['mean_squared_error'])
                        #     with col2:
                        #         st.subheader('mean_squared_error')
                        #         st.dataframe(model_compare_clf['mean_squared_error'])
                        # ray.shutdown() # 머신러닝 모델 분산 학습 종료
           
            if option == '이상탐지':
                st.subheader('머신러닝 학습 결과')
                with st.spinner('Wait for it...'):
                    if updated_df is None:
                        updated_df = df
                    json_data = updated_df.to_json() # pandas DataFrame를 json 형태로 변환
                    data = json.loads(json_data) # json을 파이썬 객체로 변환
                    
                    response = requests.post('http://127.0.0.1:8001/anomaly', json=data)  

                    if response.status_code == 200:
                        # st.write('정상 데이터 평균 점수가 낮은 순서로 추천')
                        updated_json = response.json() # NIPA 서버에서 학습한 데이터를 json으로 response 
                        # st.write(updated_json)
                        zero_df = pd.DataFrame(updated_json['json_zero'])
                        zero_df = zero_df.sort_values('novelty_mean_score', ascending=True)
                        one_df = pd.DataFrame(updated_json['json_one'])
                        one_df = one_df.sort_values('anomaly_mean_score', ascending=True)
                        
                        col1, col2 = st.columns(2) # 이상탐지 모델 결과 시각화
                        with col1:
                            st.write('정상 데이터', zero_df)
                        with col2:
                            st.write('이상 데이터', one_df)
                        score_df = pd.DataFrame(updated_json['json_score'])
                        st.write('평균 점수가 낮은 순서로 시각화')
                        for idx, model_name in enumerate(zero_df['model_name']): 
                            st.write(model_name)
                            st.line_chart(score_df[model_name])

                    # end_time = time.time()
                    # execution_time = end_time - start_time
                    # minutes, seconds = divmod(execution_time, 60)
                    # print(f"코드 실행 시간: {int(minutes)}분 {seconds:.2f}초")
                    ray.shutdown() # 머신러닝 모델 분산 학습 종료