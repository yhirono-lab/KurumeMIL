import csv
import os

def readCSV(filepath):
    csv_data = open(filepath)
    reader = csv.reader(csv_data)
    file_data = []
    header = next(reader)
    for row in reader:
        file_data.append(row)
    csv_data.close()
    return file_data

DATA_PATH = '/Raw/Kurume_Dataset' # 画像があるディレクトリへのパス
SAVE_PATH = '/Dataset/Kurume_Dataset/hirono' # 保存先のパス

txt_data = readCSV(f'/Dataset/Kurume_Dataset/csv_data/Data_SimpleName_svs.csv')
svs_tn_list = [t[0] for t in txt_data if int(t[0])<180661]
svs_fn_list = os.listdir(f'{DATA_PATH}/svs')
svs_fn_list = [svs_fn for svs_fn in svs_fn_list if int(svs_fn[4:10])<180661]
print(svs_fn_list)

for slideID in svs_tn_list:
    svs_fn = [s for s in svs_fn_list if slideID in s]
    if len(svs_fn) == 0 or len(svs_fn) == 2:
        print(slideID,svs_fn)
        