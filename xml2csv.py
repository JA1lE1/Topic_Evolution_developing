
'''
author:ja1le1
time: 2019-6-21 18:57:57
参考：http://blog.appliedinformaticsinc.com/how-to-parse-and-convert-xml-to-csv-using-python/
用途:将具有规则的爬取结果文件xml转成csv

'''

import xml.etree.ElementTree as ET
import csv

# 只需修改xml文件的路径名即可
tree = ET.parse("cn_sina.xml")
root = tree.getroot()

# open a file for writing

cn_sina_data = open('cn_sina.csv', 'w', encoding='utf-8')

# create the csv writer object

csvwriter = csv.writer(cn_sina_data)
cn_sina_data_head = []

count = 0
for member in root.findall('RECORD'):
    cn_sina = []
    # title_list = []
    if count == 0:
        id = member.find('id').tag
        cn_sina_data_head.append(id)
        iid = member.find('iid').tag
        cn_sina_data_head.append(iid)
        urls = member.find('urls').tag
        cn_sina_data_head.append(urls)

        title = member.find('title').tag
        cn_sina_data_head.append(title)

        website = member.find('website').tag
        cn_sina_data_head.append(website)

        times = member.find('times').tag
        cn_sina_data_head.append(times)

        timestamp1 = member.find('timestamp1').tag
        cn_sina_data_head.append(timestamp1)

        abstract = member.find('abstract').tag
        cn_sina_data_head.append(abstract)

        fulltext1 = member.find('fulltext1').tag
        cn_sina_data_head.append(fulltext1)

        csvwriter.writerow(cn_sina_data_head)
        count = count + 1
    id = member.find('id').text
    cn_sina.append(id)
    iid = member.find('iid').text
    cn_sina.append(iid)
    urls = member.find('urls').text
    cn_sina.append(urls)
    title = member.find('title').text
    cn_sina.append(title)
    website = member.find('website').text
    cn_sina.append(website)
    times = member.find('times').text
    cn_sina.append(times)
    timestamp1 = member.find('timestamp1').text
    cn_sina.append(timestamp1)
    abstract = member.find('abstract').text
    cn_sina.append(abstract)
    fulltext1 = member.find('fulltext1').text
    cn_sina.append(fulltext1)
    csvwriter.writerow(cn_sina)
cn_sina_data.close()