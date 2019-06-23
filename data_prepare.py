import pandas as pd

def news_prepare(file_name):
    '''
    数据准备
    file_name : csv文件名
    返回形式是yield，返回一篇的内容

    '''
    
    news_data = pd.read_csv(file_name)
    # 删除含有NaN值的任何Row，防止数据处理时的bug
    news_data = news_data.dropna()
    # news_data = news_data.dropna()
    # 对pd表重新排序
    news_data.reset_index(drop=True, inplace=True)
    for i in range(len(news_data)):
        yield news_data.loc[i, 'fulltext1']


    # yield text
