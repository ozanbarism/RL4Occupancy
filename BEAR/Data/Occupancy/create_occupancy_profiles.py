# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:57:33 2024

@author: omulayim
This script creates occupancy profiles for ApartmentHighRise and AparmentMidRise existing in BEAR environment
"""
import xarray as xr
import pandas as pd
import gc
import numpy as np
import random
import os  # Imp
# File names
file_names = ['Jan_clean.nc', 'Feb_clean.nc', 'Mar_clean.nc', 'Apr_clean.nc', 'May_clean.nc', 
              'Jun_clean.nc', 'Jul_clean.nc', 'Aug_clean.nc', 'Sep_clean.nc', 'Oct_clean.nc', 
              'Nov_clean.nc', 'Dec_clean.nc']

# Initialize an empty DataFrame to store the final dataset
df_list = []  # Use a list to collect dataframes

def process_file(file_name):
    with xr.open_dataset(file_name) as data:
        df = data.to_dataframe().reset_index()
    
    # Ensure 'time' is a datetime column
    df['time'] = pd.to_datetime(df['time'])
    
    # Return processed dataframe
    return df


# Process each file and collect the dataframes
for file_name in file_names:
    processed_df = process_file(file_name)
    df_list.append(processed_df)

# Concatenate all dataframes into a single dataframe
df = pd.concat(df_list, ignore_index=True)

# Define motion columns
motion_columns = [
    'Thermostat_DetectedMotion',
    'RemoteSensor1_DetectedMotion',
    'RemoteSensor2_DetectedMotion',
    'RemoteSensor3_DetectedMotion',
    'RemoteSensor4_DetectedMotion',
    'RemoteSensor5_DetectedMotion'
]

# Replace NaNs in motion_columns with 0s
df[motion_columns] = df[motion_columns].fillna(0)

# Compute the 'occupancy' column as the maximum of motion_columns for each row
df['occupancy'] = df[motion_columns].max(axis=1)

# Group by 'id' to organize data by house id
grouped_df = df.groupby('id')

# Create a dictionary to store each house id's data
house_data_dict = {}

for name, group in grouped_df:
    # Extract 'time' and 'occupancy' columns, and convert to DataFrame
    house_data = group[['time', 'occupancy']].copy()
    # Store in dictionary with house id as key
    house_data_dict[name] = house_data

# Optionally, clean up to free memory
del df_list, grouped_df
gc.collect()


#%%

# Step 1: Read the 'meta_data' CSV file
meta_df = pd.read_csv('meta_data.csv')

# Initialize a dictionary to store the metadata for houses in houses_with_five_sensors
meta_houses = {}

# Dictionary to store house IDs by number of occupants
occupants_dict = {}

# Step 2: Check conditions and store data
for house_id in df['id'].unique():
    # Find the corresponding row in meta_df for this house
    # Assume 'filename' column in meta_df matches house_id after stripping '.csv'
    house_meta = meta_df[meta_df['filename'].str.replace('.csv', '') == house_id]

    # If there is at least one match, continue processing
    if not house_meta.empty:
        row = house_meta.iloc[0]  # Get the matching row as a Series

        # Update the metadata dictionary
        meta_houses[house_id] = {
            'ProvinceState': row['ProvinceState'],
            'City': row['City'],
            'Floor Area [ft2]': row['Floor Area [ft2]'],
            'Number of Floors': row['Number of Floors'],
            'Age of Home [years]': row['Age of Home [years]'],
            'Number of Occupants': row['Number of Occupants'],
            'Number of Remote Sensors': row['Number of Remote Sensors'],
            'eco+ slider level': row['eco+ slider level']
        }

        # Populate the occupants dictionary
        occupants = row['Number of Occupants']
        if occupants in occupants_dict:
            occupants_dict[occupants].append(house_id)
        else:
            occupants_dict[occupants] = [house_id]

# Print or handle the occupants_dict as needed
print("House IDs grouped by number of occupants:", occupants_dict)

#%%


def generate_occupancy_data(zones, occupants_range, apartment_name, house_data_dict, occupants_dict):
    # Initialize data structures for sampled data and metadata
    occupancy_data = {}
    metadata = []
    directory_path = f'{apartment_name}'  # Directory to save CSV files
    metadata_filename = f'{directory_path}/zone_metadata.csv'
    occupancy_filename = f'{directory_path}/output_byRoom.csv'

    # Ensure the directory exists
    os.makedirs(directory_path, exist_ok=True)  # Create directory if it does not exist

    # Iterate over each zone
    for zone in zones:
        if 'CORRIDOR' in zone:
            # Corridors have zero occupancy
            occupancy_data[zone] = [0] * len(house_data_dict[next(iter(house_data_dict))]['occupancy'])
            metadata.append({'Zone': zone, 'House ID': None, 'Number of Occupants': 0})
        else:
            # Sample a number of occupants within the given range
            occupants = random.randint(*occupants_range)
            # Get a house ID from the corresponding number of occupants
            house_id = None
            if occupants in occupants_dict and occupants_dict[occupants]:
                house_id = random.choice(occupants_dict[occupants])
                # Get the occupancy data from house_data_dict
                occupancy_data[zone] = house_data_dict[house_id]['occupancy'].tolist()
            else:
                # If no house matches the occupants number, fill with zero
                occupancy_data[zone] = [0] * len(house_data_dict[next(iter(house_data_dict))]['occupancy'])
            metadata.append({'Zone': zone, 'House ID': house_id, 'Number of Occupants': occupants if house_id else 0})

    # Get the time column from any of the houses and format it
    time_column = pd.to_datetime(house_data_dict[next(iter(house_data_dict))]['time'], format='%m/%d %H:%M:%S', errors='coerce')

    # Create the occupancy DataFrame with Time as the first column
    occupancy_df = pd.DataFrame({'Time': time_column})
    for zone in zones:
        occupancy_df[zone] = occupancy_data.get(zone, [0] * len(time_column))  # Use get to handle missing zones

    # Save the occupancy DataFrame to CSV
    occupancy_df.to_csv(occupancy_filename, index=False)

    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata)

    # Save metadata to a CSV file
    metadata_df.to_csv(metadata_filename, index=False)

    # Print the output path for user confirmation
    print(f"Metadata for each zone has been saved to '{metadata_filename}'.")
    print(f"Occupancy data by room has been saved to '{occupancy_filename}'.")
    return occupancy_df



#%%
zones = [
    'G SW APARTMENT', 'G NW APARTMENT', 'OFFICE', 'G NE APARTMENT', 'G N1 APARTMENT', 'G N2 APARTMENT',
    'G S1 APARTMENT', 'G S2 APARTMENT', 'G CORRIDOR', 'M SW APARTMENT', 'M NW APARTMENT', 'M SE APARTMENT',
    'M NE APARTMENT', 'M N1 APARTMENT', 'M N2 APARTMENT', 'M S1 APARTMENT', 'M S2 APARTMENT', 'M CORRIDOR',
    'T SW APARTMENT', 'T NW APARTMENT', 'T SE APARTMENT', 'T NE APARTMENT', 'T N1 APARTMENT', 'T N2 APARTMENT',
    'T S1 APARTMENT', 'T S2 APARTMENT', 'T CORRIDOR'
]

occupants_range = (2, 3)  # Range of number of occupants
apartment_name = 'ApartmentMidRise'

# Call the function
occupancy_df = generate_occupancy_data(zones, occupants_range, apartment_name, house_data_dict, occupants_dict)
print(occupancy_df.head())


#%%
# New list of zones as provided
zones = [
    'G SW APARTMENT', 'G NW APARTMENT', 'OFFICE', 'G NE APARTMENT', 'G N1 APARTMENT', 'G N2 APARTMENT',
    'G S1 APARTMENT', 'G S2 APARTMENT', 'G CORRIDOR', 'F2 SW APARTMENT', 'F2 NW APARTMENT', 'F2 SE APARTMENT',
    'F2 NE APARTMENT', 'F2 N1 APARTMENT', 'F2 N2 APARTMENT', 'F2 S1 APARTMENT', 'F2 S2 APARTMENT', 'F2 CORRIDOR',
    'F3 SW APARTMENT', 'F3 NW APARTMENT', 'F3 SE APARTMENT', 'F3 NE APARTMENT', 'F3 N1 APARTMENT', 'F3 N2 APARTMENT',
    'F3 S1 APARTMENT', 'F3 S2 APARTMENT', 'F3 CORRIDOR', 'F4 SW APARTMENT', 'F4 NW APARTMENT', 'F4 SE APARTMENT',
    'F4 NE APARTMENT', 'F4 N1 APARTMENT', 'F4 N2 APARTMENT', 'F4 S1 APARTMENT', 'F4 S2 APARTMENT', 'F4 CORRIDOR',
    'M SW APARTMENT', 'M NW APARTMENT', 'M SE APARTMENT', 'M NE APARTMENT', 'M N1 APARTMENT', 'M N2 APARTMENT',
    'M S1 APARTMENT', 'M S2 APARTMENT', 'M CORRIDOR', 'F6 SW APARTMENT', 'F6 NW APARTMENT', 'F6 SE APARTMENT',
    'F6 NE APARTMENT', 'F6 N1 APARTMENT', 'F6 N2 APARTMENT', 'F6 S1 APARTMENT', 'F6 S2 APARTMENT', 'F6 CORRIDOR',
    'F7 SW APARTMENT', 'F7 NW APARTMENT', 'F7 SE APARTMENT', 'F7 NE APARTMENT', 'F7 N1 APARTMENT', 'F7 N2 APARTMENT',
    'F7 S1 APARTMENT', 'F7 S2 APARTMENT', 'F7 CORRIDOR', 'F8 SW APARTMENT', 'F8 NW APARTMENT', 'F8 SE APARTMENT',
    'F8 NE APARTMENT', 'F8 N1 APARTMENT', 'F8 N2 APARTMENT', 'F8 S1 APARTMENT', 'F8 S2 APARTMENT', 'F8 CORRIDOR',
    'F9 SW APARTMENT', 'F9 NW APARTMENT', 'F9 SE APARTMENT', 'F9 NE APARTMENT', 'F9 N1 APARTMENT', 'F9 N2 APARTMENT',
    'F9 S1 APARTMENT', 'F9 S2 APARTMENT', 'F9 CORRIDOR', 'T SW APARTMENT', 'T NW APARTMENT', 'T SE APARTMENT',
    'T NE APARTMENT', 'T N1 APARTMENT', 'T N2 APARTMENT', 'T S1 APARTMENT', 'T S2 APARTMENT', 'T CORRIDOR'
]
occupants_range = (2, 3)  # Range of number of occupants
apartment_name = 'ApartmentHighRise'

# Call the function
occupancy_df = generate_occupancy_data(zones, occupants_range, apartment_name, house_data_dict, occupants_dict)
print(occupancy_df.head())
