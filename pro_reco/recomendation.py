# 대시보드 UI
# https://docs.streamlit.io/knowledge-base/deploy/increase-file-uploader-limit-streamlit-cloud
# https://docs.streamlit.io/library/advanced-features/configuration#set-configuration-options
# 실행방법1: streamlit run recomendation.py
# 실행방법2: streamlit run recomendation.py --server.maxUploadSize 500 --server.maxMessageSize 500 (업로드 파일 용량 증대할 경우)
# import time # 코드 실행 시간 측정 시 사용
# sqlite:///./database/database.db
# AxiosError: Request failed with status code 403 in streamlit 발생 시, enableXsrfProtection 입력하여 실행
# streamlit run recomendation.py --server.enableXsrfProtection false

# streamlit 1.24.0 이상 버전에서 파일 업로드할 경우 AxiosError: Request failed with status code 403 발생할 수 있음
# AxiosError 403 에러 발생 시 streamlit==1.24.0 버전으로 변경 
# pip install streamlit==1.24.0

# import ray
import json
import requests
import pandas as pd
import streamlit as st
from lib.template import Template
from lib.prepro import Preprocessing
from database.connector import Database # , SelectTB
from io import StringIO
import time

# import matplotlib.pyplot as plt
# import seaborn as sns


st.set_page_config(layout="wide")
st.sidebar.title("Details")

# 분류, 이상 탐지 등 추천받을 머신러닝 모델 선택
option = st.sidebar.selectbox(
    '머신러닝 유형 선택', ('분류', '이상탐지', '회귀'))
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
            st.dataframe(updated_df, use_container_width=True)
        
        if st.sidebar.button("초기화"):
            st.cache_resource.clear()


#################### Starting ML traning
        button_for_training = st.sidebar.button("머신러닝 테스트 실행", key="button1") 
        if button_for_training: # 분류, 이상탐지 옵션에 따라 머신러닝 학습 진행
            start_time = time.time()
            # start_time = time.time() # 학습 시간 체크 시 설정


            if option == '분류':
                st.subheader('머신러닝 학습 결과')
                with st.spinner('Wait for it...'):
                    if updated_df is None:
                        updated_df = df   

                    json_data = updated_df.to_json() # pandas DataFrame를 json 형태로 변환
                    data_dump = json.dumps({'json_data':json_data, 'target': target_feture}) # 학습 데이터, Target Data 객체를 문자열로 직렬화(serialize)
                    data = json.loads(data_dump) # json을 파이썬 객체로 변환
                    response = requests.post('http://127.0.0.1:8001/new_clf', json=data) 
                    if response.status_code == 200: 
                        json_data = response.json() 
                        data = json.loads(json_data['result'])

                        #     0. accuracy 1. recall 2. precision 3. f1_weighted
                        accuracy_best_df = pd.read_json(StringIO(data['0']['best']))
                        accuracy_trial_df = pd.read_json(StringIO(data['0']['trial']))

                        recall_best_df = pd.read_json(StringIO(data['1']['best']))
                        recall_trial_df = pd.read_json(StringIO(data['1']['trial']))
                        
                        precision_best_df = pd.read_json(StringIO(data['2']['best']))
                        precision_trial_df = pd.read_json(StringIO(data['2']['trial']))

                        f1score_best_df = pd.read_json(StringIO(data['3']['best']))
                        f1score_trial_df = pd.read_json(StringIO(data['3']['trial']))

                        concat_df = pd.concat([accuracy_best_df, recall_best_df, precision_best_df, f1score_best_df], axis=1)
                        sorted_concat_df = concat_df.sort_values(by='f1_weighted', ascending=False)

                        sorted_concat_df.to_csv('./sorted_concat_df.csv', index=False)
                        accuracy_trial_df.to_csv('./accuracy_trial_df.csv', index=False)
                        recall_trial_df.to_csv('./recall_trial_df.csv', index=False)
                        precision_trial_df.to_csv('./precision_trial_df.csv', index=False)
                        f1score_trial_df.to_csv('./f1score_trial_df.csv', index=False)

                        col1, col2 = st.columns(2)
                        with col1:
                            # st.scatter_chart(sorted_concat_df.set_index('accuracy'))
                            st.scatter_chart(sorted_concat_df)
                        with col2:
                            st.dataframe(sorted_concat_df, use_container_width=True)

                        template.print_best_result(
                            'Best_accuracy', 'Best_recall', 'Best_precision', 'Best_f1_score',
                            accuracy_best_df.sort_values(by='accuracy', ascending=False), 
                            recall_best_df.sort_values(by='recall', ascending=False), 
                            precision_best_df.sort_values(by='precision', ascending=False), 
                            f1score_best_df.sort_values(by='f1_weighted', ascending=False)
                        )

                        template.print_trial_result(
                            'Trial_accuracy', 'Trial_recall', 'Trial_precision', 'Trial_f1_score',
                            accuracy_trial_df, 
                            recall_trial_df, 
                            precision_trial_df, 
                            f1score_trial_df
                        )

                        end_time = time.time()
                        execution_time = end_time - start_time  # 실행 시간 계산
                        print(f"코드 실행 시간: {execution_time} 초")
                    else:
                        st.write("Error:", response.status_code)
          
                        
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

                        # st.write(data)

                        mse_best_df = pd.read_json(StringIO(data['0']['best'])) 
                        mse_trial_df = pd.read_json(StringIO(data['0']['trial'])) 
                        mae_best_df = pd.read_json(StringIO(data['1']['best'])) 
                        mae_trial_df = pd.read_json(StringIO(data['1']['trial'])) 

                        concat_reg_df = pd.concat([mse_best_df, mae_best_df], axis=1)
                        sorted_concat_reg_df = concat_reg_df.sort_values('neg_mean_squared_error', ascending=False)

                        sorted_concat_reg_df.to_csv('./sorted_concat_reg_df.csv', index=False)
                        mse_trial_df.to_csv('./mse_trial_df.csv', index=False)
                        mae_trial_df.to_csv('./mae_trial_df.csv', index=False)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.scatter_chart(concat_reg_df)
                        with col2:
                            st.dataframe(concat_reg_df, use_container_width=True)

                        template.print_reg_best_result(
                            'Best_mean_squared_error', 'Best_mean_absolute_error',
                            mse_best_df.sort_values(by='neg_mean_squared_error', ascending=False), 
                            mae_best_df.sort_values(by='neg_mean_absolute_error', ascending=False)
                        )

                        template.print_reg_trial_result(
                            'Trial_mean_squared_error', 'Trial_mean_absolute_error',
                            mse_trial_df, mae_trial_df
                        )
                        end_time = time.time()
                        execution_time = end_time - start_time  # 실행 시간 계산
                        print(f"코드 실행 시간: {execution_time} 초")
                    else:
                        st.write("Error:", response.status_code)
           
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