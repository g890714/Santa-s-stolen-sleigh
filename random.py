'''
@auther: Chien
'''
import pandas as pd
import numpy as np
from timeit import default_timer as timer

# Evaluation Function =================================================================================================
# Idea: Convert this function into a function that takes a single array of 
#lat and a single vector of lon (length N) and returns a matrix N x N with all
# pairwise distances.
def haversine_np(lat1, lon1, lat2, lon2):
    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    
    # calculate haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    d = np.sin(dlat * 0.5)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon * 0.5)**2
    c = 2.0 * 6371.0
    return c * np.arcsin(np.sqrt(d))

def weighted_reindeer_weariness(all_trips,weight_limit = 1010, sleigh_weight = 10):
    # List out all unique trip id
    u_trip_id = all_trips["TripId"].unique()
    # Set up empty list for start & end lats & lons
    st_lats, ed_lats = [], []
    st_lons, ed_lons = [], []    
    weighted_trip_length = []   # Set up a list for each trip

    for t in u_trip_id:
        # Loop across all unique trips
        trips = all_trips[all_trips["TripId"] == t]
        # Sum all the weight of each unique trip
        weight_sum = trips["Weight"].sum() + sleigh_weight
        if weight_sum > weight_limit:
            raise Exception("One of the sleighs over weight limit!") 

        dist = 0.0
        # concatenate north pole latitude with all_trips latitude
        lats = np.concatenate([[90], trips["Latitude"], [90]])
        # concatenate north pole longitude with all_trips longitude
        lons = np.concatenate([[0], trips["Longitude"], [0]])
        # append lats and lons into st_lats, ed_lats, st_lons, ed_lons (np array)
        # append index 0 lats until one before last lats
        st_lats.append(lats[:-1]) 
        # append index 1 lats until last lats
        ed_lats.append(lats[1:])
        # append index 0 lons until one before last lons
        st_lons.append(lons[:-1])
        # append index 1 lons until last lons
        ed_lons.append(lons[1:])

        weights = [weight_sum]
        # The weight of each trip
        prev_weight = weight_sum
        # Calculating all weight reduced for each gifts dropped in each trip during returning to NP
        for weight in trips["Weight"]:
            prev_weight -= weight
            weights.append(prev_weight) 
        # Append weighted_trip_length into list so that it can be concatenated into a np array later
        weighted_trip_length.append(weights)# into a list
    # Concatenating two np array into a matrix to calculate distance between two points
    st_lats, ed_lats = np.concatenate(st_lats), np.concatenate(ed_lats)
    st_lons, ed_lons = np.concatenate(st_lons), np.concatenate(ed_lons)
    # Concatenating all weighted_trip_length into one np array
    weighted_trip_length = np.concatenate(weighted_trip_length)
    # Calculate all the distances between start and end location
    dist = haversine_np(ed_lats, ed_lons, st_lats, st_lons)
    # Compute wrw 
    wrw = np.sum(dist * weighted_trip_length)
    return wrw

# Algorithm 1: Random Search (Q1) =================================================================================================
def random_sampling(N, df):
    # Read gifts.csv and create random sample dataset(10,100,1000)
    sampledf = df.sample(n = N, replace=False)
    return sampledf

def random_search(N):
    df = pd.read_csv('gifts.csv')
    best_wrw = []
    iterL = 5    
    start_time = timer()
    for i in range(30): # Loop for 30 times to get different optimal WRW
        all_trips = random_sampling(N,df)
        new_wrw = None
        for j in range(iterL*N): # Define number of evaluations
            # Create trip id for every loop for each dataset
            all_trips["TripId"] = np.random.randint(1, N, size = N)
            wrw = weighted_reindeer_weariness(all_trips)
            if new_wrw is None:
                new_wrw = wrw
            elif wrw < new_wrw:
                new_wrw = wrw
        best_wrw.append(new_wrw)
    end_time = timer()
    best_wrw = np.array(best_wrw)
    
    # Summary Table
    print(best_wrw)
    print('Total Time Spend: ({:.2f} seconds)'.format(end_time - start_time))
    print("Best WRW:", np.min(best_wrw))
    print("Min WRW:", np.min(best_wrw))
    print("Max WRW:", np.max(best_wrw))
    print("Mean WRW:", np.mean(best_wrw))
    print("Std WRW:", np.std(best_wrw))

# Initiate Random Search Function
random_search(1000)