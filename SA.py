'''
@auther: IT puppy
'''
import pandas as pd
import numpy as np
import random
from timeit import default_timer as timer

#pick data randomly
def random_sampleling(df, num, i):
    np.random.seed = (30+i)
    subset = df.sample(n=num, replace=False)
    return subset

def haversine_np(lat1, lon1, lat2, lon2):
    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))    
    # calculate haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    d = np.sin(dlat * 0.5)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon * 0.5)**2
    c = 2.0 * 6371.0
    return c * np.arcsin(np.sqrt(d))

#to get weighted reindeer weariness
def getwrw(all_trips, weight_limit = 1010, sleigh_weight = 10):
    trip_ids = all_trips["TripId"].unique()
    
    # st_location and ed_location in list
    st_lats, ed_lats = list(), list()   
    st_lons, ed_lons = list(), list()
    
    total_wts = list()
    for trip_id in trip_ids:   
        trips = all_trips[all_trips["TripId"]==trip_id]
        weight_sum = trips["Weight"].sum()+sleigh_weight
        lats = np.concatenate([[90], trips["Latitude"], [90]])
        lons = np.concatenate([[0], trips["Longitude"], [0]])

        st_lats.append(lats[:-1])
        ed_lats.append(lats[1:])
        st_lons.append(lons[:-1])
        ed_lons.append(lons[1:])
        weights = [weight_sum]
        
        # The weight of each trip
        prev_weight = weight_sum
        for weight in trips["Weight"]:
            prev_weight -= weight
            weights.append(prev_weight) 
        total_wts.append(weights)
        
    st_lats, ed_lats = np.concatenate(st_lats), np.concatenate(ed_lats)
    st_lons, ed_lons = np.concatenate(st_lons), np.concatenate(ed_lons)
    total_wts = np.concatenate(total_wts)
    # calculate all the distances between st_location and ed_location
    distances = haversine_np(ed_lats, ed_lons, st_lats, st_lons)
    # final wrw
    wrw = np.sum(distances*total_wts)
    return wrw

def nm2(df):
    group_trip = df.groupby('TripId')
    while True:
        # random select trip
        rd_trip = np.random.choice(df['TripId'].unique(),1) # random select a trip to go through swapping
        sl_trip = list()
        sl_trip.append(group_trip.get_group(rd_trip[0])) # find the group with TripId = sp_trip[0]
        sl_trip = sl_trip[0].reset_index(drop=True) # first(sp_trip[0]) dataframe in sl_trip
        
        sl_gift = np.random.choice(sl_trip.index.tolist(),1)    # within trip, random select gift and position
        g = sl_gift[0]
        sl_position = np.random.choice(sl_trip.index.tolist(),1)
        p = sl_position[0]
    
        # let's change!    
        if g > p:
            temp = sl_trip.iloc[p:g].copy()
            sl_trip.iloc[p] = sl_trip.iloc[g]        
            if g == len(sl_trip)-1:
                temp.index = range(p+1, g+1, 1)
                sl_trip.iloc[p+1:] = temp
                break
            else:
                temp.index = range(p+1, g+1, 1)
                sl_trip.iloc[p+1:g+1] = temp
                break         
        elif g < p:
            if p == len(sl_trip)-1:
                temp = sl_trip.iloc[g+1:].copy()
            else:
                temp = sl_trip.iloc[g+1:p+1].copy()        
            sl_trip.iloc[p] = sl_trip.iloc[g]
            temp.index = range(g, p, 1)
            sl_trip.iloc[g:p] = temp
            break                
        else:
            continue    
    rest_trip_df = df[df['TripId'] != rd_trip[0]]
    new_all_df = rest_trip_df.append(sl_trip)    
    return new_all_df

def Simulated_Annealing(df, t, best_wrw):
    new_df = nm2(df)
    new_wrw = getwrw(new_df)
    p = np.exp((best_wrw-new_wrw)/(best_wrw*t))
    ran = random.random()
    if new_wrw < best_wrw:
        best_wrw = new_wrw
        return best_wrw, new_df
    else:
        if ran < p:
            best_wrw = new_wrw
            return best_wrw, new_df
        else:
            return best_wrw, new_df    

def main(subset_num, data, times=30):
    time = list()
    wrw_arr = list()
    t = 1   #initial temperature
    delta = 0.98    #cooling factor

    for i in range(times): #times
        s_time = timer()
        all_trips = random_sampleling(data, subset_num, i)
        while True:
            all_trips["TripId"] = np.random.randint(1, subset_num, size = subset_num)
            a = all_trips.groupby('TripId')['Weight']
            b = a.agg([np.sum])
            length = len(all_trips['TripId'].unique())
            for i in range(length):
                if b['sum'].iloc[i] > 1000:
                    break
                else:
                    continue
            control = 1
            if control == 1:
                break
        for j in range(1000):
            if j==0:
                best_wrw = float('inf')
            else:
                best_wrw, all_trips = Simulated_Annealing(all_trips, t, best_wrw)                   
            t = t*delta
        e_time = timer()
        inner = e_time - s_time
        print('times takes:',inner)
        time.append(inner)
        wrw_arr.append(best_wrw)
    wrw_arr = np.array(wrw_arr)
    # summary
    print(time)
    print(wrw_arr)
    print("Minimum: %f"%wrw_arr.min())
    print("Maximum: %f"%wrw_arr.max())
    print("Mean: %f"%wrw_arr.mean())
    print("Std: %f"%wrw_arr.std())

gifts = pd.read_csv('gifts.csv')
num = subset_num = input("Please input the subset number: ")
num = int(num)
start_time = timer()
main(num, gifts)
end_time = timer()

print('Total Time Spend: ({:.2f} seconds)'.format(end_time - start_time))