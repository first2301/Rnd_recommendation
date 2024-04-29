import sqlite3
import pandas as pd

db_path = './database.db'
conn = sqlite3.connect(db_path)

df1 = pd.DataFrame({'table1_1': range(6),
                   'table1_2': range(6),
                   'table1_3': range(6),})

df2 = pd.DataFrame({'table2_1': range(6),
                   'table2_2': range(6),
                   'table2_3': range(6),})

df3 = pd.DataFrame({'table3_1': range(6),
                   'table3_2': range(6),
                   'table3_3': range(6),})
print('df1')
print(df1)

print('df2')
print(df2)

print('df3')
print(df3)

df1.to_sql('test1_db', conn, if_exists='replace', index=False)
df2.to_sql('test2_db', conn, if_exists='replace', index=False)
df3.to_sql('test3_db', conn, if_exists='replace', index=False)

conn.close()