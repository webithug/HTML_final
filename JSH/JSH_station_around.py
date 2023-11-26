import os
import json
import numpy as np
import copy
import os
import csv
from math import radians, sin, cos, sqrt, atan2



#兩點之間距離     #單位：公尺
def haversine_distance(lat1, lon1, lat2, lon2):
    # 將經緯度轉換為弧度
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine 公式計算距離
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371 * c  # 地球半徑為 6371 公里

    return distance*1000    #單位：公尺

#找出目標staion附近有哪些station     #distance_threshold 單位：公尺
def find_points_within_distance(target_lat, target_lon, points, distance_threshold, station_numbers):
    nearby_points = []     #給定範圍內的站編號

    for i in range(len(points)):
        point_lat, point_lon = points[i]
        distance = haversine_distance(target_lat, target_lon, point_lat, point_lon)

        if distance <= distance_threshold:
            nearby_points.append(station_numbers[i])
    return nearby_points


path = "/Users/web/Library/CloudStorage/Dropbox/NTU_course/Senior1/ML_HT/code/final_regular"

#取得目前有的資料dates
dates = []
# 取得資料夾中所有檔案的名稱
# 使用 os 模組的 listdir 方法列出資料夾中的所有檔案和子資料夾
file_names = os.listdir(path+"/html.2023.final.data/release")
for file_name in file_names:
    dates.append(file_name)
date = sorted(dates,reverse=False)

#所有時間的string ['00:00','00:01',...,'23:59']
time = []
for hr in range(24):
    for min in range(60):
        time.append(str(f"{hr:02d}")+ ":"+ str(f"{min:02d}"))


d = open(path+"/html.2023.final.data/demographic.json",encoding="utf-8")
demographic = json.load(d)
d.close

station_numbers = list(demographic.keys())    #全部的
print("key_done")
target_station = ['500101001', '500101002', '500101003', '500101004', '500101005', '500101006', '500101007', '500101008', '500101009', '500101010', '500101013', '500101014', '500101015', '500101018', '500101019', '500101020', '500101021', '500101022', '500101023', '500101024', '500101025', '500101026', '500101027', '500101028', '500101029', '500101030', '500101031', '500101032', '500101033', '500101034', '500101035', '500101036', '500101037', '500101038', '500101039', '500101040', '500101041', '500101042', '500101091', '500101092', '500101093', '500101094', '500101114', '500101115', '500101123', '500101166', '500101175', '500101176', '500101181', '500101184', '500101185', '500101188', '500101189', '500101190', '500101191', '500101193', '500101199', '500101209', '500101216', '500101219', '500105066', '500106002', '500106003', '500106004', '500119043', '500119044', '500119045', '500119046', '500119047', '500119048', '500119049', '500119050', '500119051', '500119052', '500119053', '500119054', '500119055', '500119056', '500119057', '500119058', '500119059', '500119060', '500119061', '500119062', '500119063', '500119064', '500119065', '500119066', '500119067', '500119068', '500119069', '500119070', '500119071', '500119072', '500119074', '500119075', '500119076', '500119077', '500119078', '500119079', '500119080', '500119081', '500119082', '500119083', '500119084', '500119085', '500119086', '500119087', '500119088', '500119089', '500119090', '500119091']

# 所有點的經緯度
points =[]
for station in station_numbers:
    points.append( (demographic[station]["lat"],demographic[station]["lng"]) )
print("long,lati done")

for target in target_station:

    # 目標點的經緯度 500101001
    target_latitude = demographic[target]["lat"]
    target_longitude = demographic[target]["lng"]

    # 指定距離閾值（公尺）
    distance_threshold = 100  #  公尺

    # 找出在指定距離內的點
    nearby_points = find_points_within_distance(target_latitude, target_longitude, points, distance_threshold, station_numbers)

    data_list = []
    for  point in nearby_points:
        
        #讀csv
        with open(path+"/JSH/train_data/"+point+".csv", 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader: 
                #row = [站編號,date,time(min 0-1440),tot,bemp,act,lat,lng,sbi]
                for i in range(len(row)-3):
                    row[i] = int(row[i])
                del row[4]     #把bemp刪掉
                data_list.append(row) 

    # 使用csv模块写入CSV文件
    with open(path+"/JSH/train_data_add_nearby/"+target+".csv", 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入数据
        writer.writerows(data_list)








