import h5py
import numpy as np
import pandas as pd
import sys

ERROR_REJECTION_DT = 0.5
ERROR_REJECTION_VEL = 10
ERROR_REJECTION_SD = 3.4
slope_coef = 940.9411
slope_pow = -.8097



f = h5py.File('../../CS229/ssi71.h5','r')

#Use sys parameters if you want to import a different time range! If not, defaults are given.
if len(sys.argv) < 4:
    print("Parameters: <output_file.csv> <start_index> <end_index> <start_time>")
    print(len(sys.argv))
    output_file = "ssi_pressure_testfin_badpressure.csv"
    start_index = 1245310#2742657#3368179 #1529169147562
    end_index = 1283940#2803171#3460119 #1529173744553
    start_time = 1529059500000#1529169147562#1529169147562
    #start_time = 1529175944249
else:
    print("sys:", sys.argv)
    output_file = sys.argv[1]
    start_index = int(sys.argv[2])
    end_index = int(sys.argv[3])
    start_time = int(sys.argv[4])

n_sensors = 4
sampling_rate = 20 #picks one in every 20 ratings

altitude_barometer = np.array(f['df'].get('block1_values')[2])[::sampling_rate]
raw_pressure = np.array(f['df'].get('block1_values')[start_index:end_index,45:49]).T[::,::sampling_rate]
time = np.array(f['df'].get('axis1')[start_index:end_index]).T[::sampling_rate]
temp = np.array(f['df'].get('block1_values')[start_index:end_index,7]).T[::sampling_rate]
press = np.array(f['df'].get('block1_values')[start_index:end_index,49]).T[::sampling_rate]

bmp1rej = np.array(f['df'].get('block5_values')[start_index:end_index,13]).T[::sampling_rate]
bmp2rej = np.array(f['df'].get('block5_values')[start_index:end_index,14]).T[::sampling_rate]
bmp3rej = np.array(f['df'].get('block5_values')[start_index:end_index,15]).T[::sampling_rate]
bmp4rej = np.array(f['df'].get('block5_values')[start_index:end_index,16]).T[::sampling_rate]
ascentrate = np.array(f['df'].get('block1_values')[start_index:end_index,4]).T[::sampling_rate]

print(np.size(temp))

last_accepted_pressure = np.zeros((4))
last_accepted_time = np.zeros((4))
#bmp_enabled = np.zeros((4))


def velocity_check(df_index, raw_pressure, time):
    if df_index == 0:
        for j in range(0, n_sensors):
            last_accepted_pressure[j] = raw_pressure[j][df_index]
            last_accepted_time[j] = time[df_index]
        return np.ones((4))

        #set defaults
    accepts = np.zeros((4))
    for i in range(0,n_sensors):
        slope = slope_coef*np.power(last_accepted_pressure[i],slope_pow)
        ts = ERROR_REJECTION_DT + (time[df_index] - last_accepted_time[i])
        #print(raw_pressure[i])
        vel = slope*(abs(raw_pressure[i][df_index]- last_accepted_pressure[i]) - ERROR_REJECTION_SD*5)/ts
        if (vel <= ERROR_REJECTION_VEL): #and bmp_enabled[i]):
            #if velocity is within threshold and sensor enabled, accept
            accepts[i] = 1
            last_accepted_time[i] = time[df_index]
            last_accepted_pressure[i] = raw_pressure[i][df_index]
            #smooth the data that is not rejected

    return accepts

# acceptedbasic = np.zeros((n_sensors, end_index - start_index))
# print(acceptedbasic)
# for k in range(0, end_index - start_index):
#     new_accepts = velocity_check(k, raw_pressure, time)
#     for j in range(0, n_sensors):
#         acceptedbasic[j][k] = new_accepts[j]
#         if np.sum(acceptedbasic[j]) < 2:


accepted = np.ones(len(time))

""" OLD SET LABELS FUNCTION """
# def set_labels(arr, start_time, a, b):
#     a_index = int((a - start_time)/(50*20))
#     b_index = int((b - start_time)/(50*20))
#     print(a_index, b_index)
#     arr[a_index:b_index] = np.zeros((b_index - a_index))
#     return(arr)

def is_bad_time(timestamp, time_ranges, shouldp):
    if shouldp:
        print(timestamp)
        # print(time_ranges[0][0])

    for time_range in time_ranges:
        if timestamp > time_range[0] and timestamp < time_range[1]:

            return True
    return False

def set_labels(arr,bad_times, time_arr):
    for i in range(len(time_arr)):
        if is_bad_time(time_arr[i],bad_times, False):
            arr[i]=0
    return arr

""" ssi_pressure_labels.csv"""
# bad_time_ranges = [
# (1529170069945000000, 1529170089030000000),
# (1529170260176000000, 1529170319689000000),
# (1529170941689000000, 1529171168560000000),
# (1529171797418000000, 1529171852313000000),
# (1529172292538000000, 1529172336343000000),
# (1529172859979000000, 1529172991451000000),
# (1529173465386000000, 1529173516745000000)
# ]


"""val set - based on altitude:"""
# bad_time_ranges = [
# (1529137374461000000, 1529137381161000000),
# (1529137703076000000, 1529137763631000000),
# (1529138276204000000, 1529138399988000000),
# (1529138631501000000, 1529138633301000000),
# (1529138686265000000, 1529138687599000000),
# (1529138910679000000, 1529138925414000000),
# (1529139410504000000, 1529139596603000000),
# (1529139824381000000, 1529139835316000000)
# ]

"""val set: based on pressure:"""
# bad_time_ranges = [
# (1529137315606000000, 1529137335956000000),
# (1529137684226000000, 1529137730979000000),
# (1529138243472000000, 1529138361764000000),
# (1529138616462000000, 1529138637401000000),
# (1529138867016000000, 1529138896740000000),
# (1529139373291000000, 1529139561532000000),
# (1529139765616000000, 1529139888808000000)
# ]

"""test set: based on pressure:"""
bad_time_ranges = [
(1529059986834000000, 1529060064462000000),
(1529060193146000000, 1529060232469000000),
(1529060392526000000, 1529060407416000000),
(1529060608350000000, 1529060669500000000),
(1529061068781000000, 1529061111354000000),
(1529061304338000000, 1529061332515000000)
]


#for i in range(0, len(bad_time_ranges)):
    #accepted = set_labels(accepted, start_time, bad_time_ranges[i][0], bad_time_ranges[i][1])
accepted = set_labels(accepted,bad_time_ranges,time)


baselinepreds = np.ones(len(time))
numCorrect = 0
for i in range(0,len(baselinepreds)):
    # diff1 = bmp1rej[i] - bmp1rej[i-1]
    # diff2 = bmp2rej[i] - bmp2rej[i-1]
    # diff3 = bmp3rej[i] - bmp3rej[i-1]
    # diff4 = bmp4rej[i] - bmp4rej[i-1]
    #
    # sum = diff1 + diff2 + diff3 + diff4
    if ascentrate[i] > 20:
        baselinepreds[i] = 0
    else:
        baselinepreds[i] = 1
    if baselinepreds[i] == accepted[i]:
        numCorrect+=1
print("Baseline correct rate: ", numCorrect/len(time))


training_set = pd.DataFrame(columns=["time", "temperature", "pressure", "press_change", "label"])
training_set["time"] = time
training_set["temperature"] = temp
training_set["pressure"] = press
training_set["press_change"] = [abs(press[i] - press[i-1]) if i > 0 else 0 for i in range(0,len(press))]
training_set["var"] = [np.var(press[j-5:j]) if j >= 5 else 0 for j in range(len(press))]

training_set["label"] = accepted

training_set.to_csv(output_file)
training_set["temp_change"] = [temp[i] - temp[i-1] if i > 0 else 0 for i in range(0,len(temp))]

#Some summary stats:
print("Counts: ", training_set.groupby("label").count()["pressure"])
print("Mean absolute pressure change over 1 second: ", training_set.apply(abs).groupby("label").mean()["press_change"])
print("SD of absolute pressure change over 1 second: ", training_set.apply(abs).groupby("label").std()["press_change"])
print("Mean absolute temperature change over 1 second: ", training_set.apply(abs).groupby("label").mean()["temp_change"])
print("SD of absolute temperature change over 1 second: ", training_set.apply(abs).groupby("label").std()["temp_change"])
print("Mean temperature: ", training_set.groupby("label").mean()["temperature"])
print("SD of temperature: ", training_set.groupby("label").std()["temperature"])

#pressure change seems like it could be a pretty salient feature, temperature not so much.

"""
FIRST SET (FOR TRAINING)
        time            index
start   1529169147562   3368179
end     1529173744553   3460119

labelling: these are *time* ranges where sensor readings are bad:
1529170069945 - 1529170089030
1529170260176 - 1529170319689
1529170941689 - 1529171168560
1529171797418 - 1529171852313
1529172292538 - 1529172336343
1529172859979 - 1529172991451
1529173465386 - 1529173516745

SECOND SET (FOR VALIDATION)
        time            index
start   1529136796191   2742657
end     1529139900309   2803171

labelling: these are *time* ranges where altitude readings are bad:
1529137374461 - 1529137381161
1529137703076 - 1529137763631
1529138276204 - 1529138399988
1529138631501 - 1529138633301                                            <<<<_
1529138686265 - 1529138687599 //from same pressure bumb as prev alt bad range-'
1529138910679 - 1529138925414
1529139410504 - 1529139596603
1529139824381 - 1529139835316

labelling: these are *time* ranges where pressure readings are bad:
1529137315606 - 1529137335956
1529137684226 - 1529137730979
1529138243472 - 1529138361764
1529138616462 - 1529138637401
1529138867016 - 1529138896740
1529139373291 - 1529139561532
1529139765616 - 1529139888808


THIRD SET (FOR TESTING)
        time            index
start   1529059500000   1245310
end     1529061500000   1283940

labelling: these are *time* ranges where pressure readings are bad:
1529059986834 - 1529060064462
1529060193146 - 1529060232469
1529060392526 - 1529060407416
1529060608350 - 1529060669500
1529061068781 - 1529061111354
1529061304338 - 1529061332515


features:
- previous X readings
- previous altitudes
- raw_temp
- press ()
"""


"""
FEATURES:

time
altitude_gps
altitude_barometer
last_accepted
last_rejected


"""
