import pandas as pd
import glob
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy import optimize as op
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import json
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from functools import partial



# function returns a dict of taipei rain data. {"20231121": 0.0, ... }
def read_rain_data(json_file_path): 
    # Open the JSON file and load the data
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    list_of_dics = data['cwaopendata']['resources']['resource']['data']['surfaceObs']['location'][3]['stationObsTimes']['stationObsTime']
    daily_rain_list = []
    for day in list_of_dics:
        date = day['Date'].replace("-", "")
        rain_mm = day['weatherElements']['Precipitation']
        
        if rain_mm == 'T':
            rain_mm = 0.25
        else:
            rain_mm = float(rain_mm)
        
        daily_rain_list.append([date, rain_mm])
    
    # Convert to a list of dictionaries
    date_rain_dict = {sublist[0]: sublist[1] for sublist in daily_rain_list }
    
    return date_rain_dict

# return list of holiday dates, input format: "2023-10-01"
def holiday(start_date, end_date):

    # Generate a date range
    date_range = pd.date_range(start=start_date, end=end_date)

    # Filter weekends (Saturday and Sunday)
    weekend_dates = date_range[date_range.weekday.isin([5, 6])]

    # List of public holidays in Taiwan (example dates)
    holidays = ["2023-10-09", "2023-10-10"]

    # Combine weekends and holidays
    all_dates = weekend_dates.union(pd.to_datetime(holidays))

    all_dates = [ date.strftime("%Y%m%d") for date in all_dates ]
    
    return all_dates
    
def what_day(date_string):
    # Parse the date string
    date_object = datetime.strptime(date_string, "%Y%m%d")

    # Get the day of the week (Monday is 0 and Sunday is 6)
    day_of_week = date_object.weekday()

    return day_of_week




def main():

    # jsh_mode on of off
    jsh_mode = 0

    test_stations_file = "/Users/web/Library/CloudStorage/Dropbox/NTU_course/Senior1/ML_HT/code/final_regular/html.2023.final.data/sno_test_set.txt"

    df = pd.read_csv(test_stations_file, header=None, names=['stationID'])
    test_stations_list = df['stationID'].tolist()
    # print(test_stations_list)

    # read the rain data
    daily_rain_dict = read_rain_data("/Users/web/Library/CloudStorage/Dropbox/NTU_course/Senior1/ML_HT/code/final_regular/C-B0025-001.json")

    # list of holidays
    holidays_list = holiday("2023-10-01", "2023-12-30")
    # print(holidays_list)

    # record the cv_score
    cv_score_list = []


    # loop over each station in sno_test_set.txt, train a model for each station
    outputlist = []
    for station in test_stations_list:
        # x format final: stationID, date, time(min 0-1440), tot, act, rain, holiday, weekday (but don't train with date, tot)
        x_train = []
        y_train = []

        # get the data of the current station
        data_file = "/Users/web/Library/CloudStorage/Dropbox/NTU_course/Senior1/ML_HT/code/final_regular/JSH/train_data_add_nearby/" + str(station) + ".csv"
        df_station_data = pd.read_csv(data_file, header=None)

        # loop over the df to save data as x_train and y_train
        for idx, row in df_station_data.iterrows():
            # x format: stationID, date, time(min 0-1440), tot, act
            x_new = row[0:5].tolist()
            x_new = [ int(entry) for entry in x_new ]
            x_train.append( x_new )

            y_new = row[7]
            y_train.append(y_new)
        
        # add rain and holiday data to x_train -> x format: stationID, date, time(min 0-1440), tot, rain, holiday, weekday
        for x in x_train:
            date_str = str(x[1])
            x.append(daily_rain_dict[date_str])
            # print(daily_rain_dict[date_str])
            if date_str in holidays_list:
                x.append(1)
                # print(x)
            else:
                x.append(0)
            # add weekday 0~6
            x.append( what_day(date_str) )
            

        
        
        # delete x_train of 10/25 to 10/28 if jsh_mode is on
        if jsh_mode == 1:
            jsh_test_date = [ 20231025, 20231026, 20231027, 20231028 ]
            # Use a list comprehension to filter out elements based on the condition
            filtered_indices = [i for i, x in enumerate(x_train) if x[0] not in jsh_test_date]

            # Create new lists with filtered elements
            x_train = [x_train[i] for i in filtered_indices]
            y_train = [y_train[i] for i in filtered_indices]
        
        


        # Choose a model and train
        model = RandomForestRegressor(n_estimators=250, random_state=1126, n_jobs=10, verbose=0)
        # model = LinearRegression()
        # dont train with date, tot
        x_train_input = [ x[2:3] + x[4:] for x in x_train ]
        # print(x_train_input)
        model.fit(x_train_input, y_train )

        tot = x_train[0][3]

        # crosss validation
        def error_funct(y_true, y_predict, s=tot):
            y_true = np.array(y_true)
            y_predict = np.array(y_predict)
            return np.mean( 3 * ( np.abs(y_predict-y_true)/s ) * ( np.abs(y_true/s-1/3) + np.abs(y_true/s-2/3) ) )
        scorer = make_scorer(error_funct, greater_is_better=False)
        cv_score = cross_val_score(model, x_train_input, y_train, scoring=scorer )
        cv_score_list.append(cv_score)
        print(station ,-cv_score)

        continue
        

        # create x_test for public test: From 10/21/2023 00:00 to 10/24/2023 23:40. (i.e. 00:00, 00:20, 00:40, … 23:00, 23:20, 23:40)
        x_test_public = []
        # print(tot)
        for date in range(20231021, 20231025):
            for time in range(1, 1440, 20):
                x_test_public_new = [station, date, time, tot, 1]
                x_test_public.append(x_test_public_new)

        # create x_test for stage 1 private test. From 12/4/2023 00:00 to 12/10/2023 23:40 
        x_test_private1 = []
        for date in range(20231204, 20231211):
            for time in range(1, 1440, 20):
                x_test_private1_new = [station, date, time, tot, 1]
                x_test_private1.append(x_test_private1_new)

        # add rain data to x_test_public
        for x in x_test_public:
            date_str = str(x[1])
            # add rain
            if date_str in daily_rain_dict:
                x.append(daily_rain_dict[date_str])
            else:
                x.append(0)
            # add holiday
            if date_str in holidays_list:
                x.append(1)
            else:
                x.append(0)
            # add weekday 0~6
            x.append( what_day(date_str) )
        
        # add rain data to x_test_private1
        for x in x_test_private1:
            date_str = str(x[1])
            # add rain
            if date_str in daily_rain_dict:
                x.append(daily_rain_dict[date_str])
            else:
                x.append(0)
            # add holiday
            if date_str in holidays_list:
                x.append(1)
            else:
                x.append(0)
            # add weekday 0~6
            x.append( what_day(date_str) )

        # create x_test for JSH test: From 10/25/2023 00:00 to 10/28/2023 23:40. (i.e. 00:00, 00:20, 00:40, … 23:00, 23:20, 23:40)
        x_test_jsh = []
        for date in range(20231025, 20231029):
            for time in range(1, 1440, 20):
                x_test_jsh_new = [station, date, time, tot, 1]
                x_test_jsh.append(x_test_jsh_new)


        # add rain data to x_test_jsh
        if jsh_mode == 1:
            for x in x_test_jsh:
                date_str = str(x[1])
                # add rain
                if date_str in daily_rain_dict:
                    x.append(daily_rain_dict[date_str])
                else:
                    x.append(0)
                # add holiday
                if date_str in holidays_list:
                    x.append(1)
                else:
                    x.append(0)
                # add weekday 0~6
                x.append( what_day(date_str) )

        
        # Make predictions on the test set, don't train with date, tot
        x_test_public_input = [ x[2:3] + x[4:]  for x in x_test_public]
        predictions_public = model.predict(x_test_public_input)

        x_test_private1_input = [ x[2:3] + x[4:]  for x in x_test_private1]
        predictions_private1 = model.predict(x_test_private1_input)

        if jsh_mode == 1:
            x_test_jsh_input = [ x[2:3] + x[4:] for x in x_test_jsh]
            predictions_jsh = model.predict(x_test_jsh_input)
        # print(x_test_public_input)
        

        # save the predictions of public to submission file
        for x, sbi in zip(x_test_public, predictions_public):
            # print(x)
            time_in_correct_format =  str( timedelta(minutes=x[2]-1) )
            
            # Parse the input time string
            time_in_correct_format = datetime.strptime(time_in_correct_format, "%H:%M:%S")
            # Format the time as HH:MM
            formatted_time = str(time_in_correct_format.strftime("%H:%M"))

            id = str(x[1]) + "_" + str(station) + "_" + formatted_time
            
            outputlist.append( [id, sbi] )

        # save the predictions of private to submission file
        for x, sbi in zip(x_test_private1, predictions_private1):
            # print(x)
            time_in_correct_format =  str( timedelta(minutes=x[2]-1) )
            
            # Parse the input time string
            time_in_correct_format = datetime.strptime(time_in_correct_format, "%H:%M:%S")
            # Format the time as HH:MM
            formatted_time = str(time_in_correct_format.strftime("%H:%M"))

            id = str(x[1]) + "_" + str(station) + "_" + formatted_time
            
            outputlist.append( [id, sbi] )

        if jsh_mode == 1:
            # save the predictions of jsh_test to submission file
            for x, sbi in zip(x_test_jsh, predictions_jsh):
                # print(x)
                time_in_correct_format =  str( timedelta(minutes=x[2]-1) )
                
                # Parse the input time string
                time_in_correct_format = datetime.strptime(time_in_correct_format, "%H:%M:%S")
                # Format the time as HH:MM
                formatted_time = str(time_in_correct_format.strftime("%H:%M"))

                id = str(x[1]) + "_" + str(station) + "_" + formatted_time
                
                outputlist.append( [id, sbi] )

        
    # # output the results to a submission file
    # output_file = '/Users/web/Library/CloudStorage/Dropbox/NTU_course/Senior1/ML_HT/code/final_regular/submission_web.csv'
    # # Save the list of lists to a text file with each list on a new line
    # column_names = ['id', 'sbi']
    # np.savetxt(output_file, outputlist, fmt='%s, %s', delimiter=',', header=','.join(column_names), comments='', newline='\n') 

    # print(- np.mean( np.array(cv_score_list) ))





if __name__ == "__main__":
    main()
    # print(read_rain_data("/Users/web/Library/CloudStorage/Dropbox/NTU_course/Senior1/ML_HT/code/final_regular/C-B0025-001.json"))