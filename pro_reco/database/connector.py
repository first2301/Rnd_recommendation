from sqlalchemy import create_engine, Table, MetaData
import pandas as pd

class Database:
    def __init__(self, path):
        self.path = path
        self.engine = create_engine(path, connect_args={"check_same_thread": False})
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)

    def connect(self):
        return self.engine.connect()
    
    def tables_info(self):
        ''' 
        데이터베이스에 있는 전체 테이블 정보 확인
        '''
        return self.metadata.tables.keys()

    def db_query(self, table_name):
        '''
        선택한 테이블 조회
        '''
        with self.connect() as conn:
            table = Table(table_name, self.metadata, autoload_with=self.engine)
            query = table.select()
            return conn.execute(query)
        
    def db_to_df(self, tb_name):
        '''
        Database table를 pandas dataframe로 변환
        '''
        with self.connect() as conn:
            table = Table(tb_name, self.metadata, autoload_with=self.engine)
            query = table.select()
            result = conn.execute(query)
            df = pd.DataFrame(result, columns=result.keys())
            return df
