import json
import requests
import pandas as pd
import streamlit as st
from database.connector import Database
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")
st.sidebar.title("Details")

option = st.sidebar.selectbox(
    'Select Machine Learning Task', ('분류', '이상탐지'))
connecton_option = st.sidebar.selectbox(
    'Select how to upload data', ('File_upload', 'DB_connection'))
uploaded_file = None
df = None
updated_df = None

# Choose File upload or DB connection
if connecton_option == 'File_upload':
    uploaded_file = st.sidebar.file_uploader("csv file upload", type="csv") # 파일 업로드
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
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)
try:
    if connecton_option == 'File_upload':
        df = load_data(uploaded_file)  
except ValueError:
    st.sidebar.write('Need to upload csv file')

    # 데이터 전처리
    if uploaded_file is not None or df is not None:
        st.subheader('데이터 분석')
        col_list = df.columns.tolist() # 데이터 전처리 옵션 설정 리스트
        target_feture = ""
        if option == '분류':
            target_feture = st.sidebar.multiselect('Select Target Column', options=col_list)

        data_to_drop = st.sidebar.multiselect('Drop Cloumns', options=col_list)

        tab_eda_df, tab_eda_info, tab_Label_counts = st.tabs(['Original data', 'Null information', 'Target Data Counts']) # tab_Label_counts Labels counts