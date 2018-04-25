import pandas as pd

writer = pd.ExcelWriter('output.xlsx')

file_name = 'simple_q_table'
df = pd.read_pickle(file_name+'.gz',compression='gzip')
df.to_excel(writer,'Sheet1')
writer.save()