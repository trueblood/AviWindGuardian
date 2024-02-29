import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import time

#create single collision for each individual wind turbine and create dataset with collision location and time and bird species

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3956  # Radius of Earth in miles
    return c * r

# Adapt the function to accept a tuple for turbine data
def check_collision(data):
    (turbine_longitude, turbine_latitude), bird_data = data
    collisions = []
    for bird in bird_data:
        if haversine(turbine_longitude, turbine_latitude, bird['Longitude'], bird['Latitude']) <= 1:
            collision_data = {
                'Species': bird['Species'],
                'Turbine_Longitude': turbine_longitude,
                'Turbine_Latitude': turbine_latitude,
                'Timestamp': bird['Timestamp']
            }
            collisions.append(collision_data)
    return collisions

start_time = time.time()

num_cores = os.cpu_count() - 1

wind_turbines_df = pd.read_csv('wind_turbine_location_20231128.csv')
bird_data_df = pd.read_csv('combined_bird_data.csv', nrows=100)

wind_turbines_df.rename(columns={'xlong': 'Longitude', 'ylat': 'Latitude'}, inplace=True)

bird_data_list = bird_data_df[['Longitude', 'Latitude', 'Species', 'Timestamp']].to_dict('records')

# Create tuples for turbine coordinates
turbine_coords = [(row['Longitude'], row['Latitude']) for index, row in wind_turbines_df.iterrows()]

args = [((longitude, latitude), bird_data_list) for longitude, latitude in turbine_coords]

detailed_collisions = []
with ProcessPoolExecutor(max_workers=num_cores) as executor:
    results = list(executor.map(check_collision, args))
    for result in results:
        detailed_collisions.extend(result)

detailed_collision_df = pd.DataFrame(detailed_collisions)
print(detailed_collision_df.head())
detailed_collision_df.to_csv('detailed_wind_turbine_collisions.csv', index=False) # this one looks just fine

collision_csv_filename = 'wind_turbines_with_collisions.csv'

# Calculate collisions per turbine location, handling the case where no collisions are found
if not detailed_collision_df.empty:
    collision_counts = detailed_collision_df.groupby(['Turbine_Longitude', 'Turbine_Latitude']).size().reset_index(name='Collision_Count')
    # Merge this summary back into the original wind_turbines_df
    # Use left merge to ensure all turbines are included, even those without collisions
    wind_turbines_df = wind_turbines_df.merge(collision_counts, how='left', left_on=['Longitude', 'Latitude'], right_on=['Turbine_Longitude', 'Turbine_Latitude'])

    # Fill NaN values in Collision_Count with 0 to indicate no collisions
    wind_turbines_df['Collision_Count'] = wind_turbines_df['Collision_Count'].fillna(0)

    # Keep only the required columns
    wind_turbines_df = wind_turbines_df[['Longitude', 'Latitude', 'Collision_Count']]

    # Save the enriched wind_turbines_df with collision counts
    wind_turbines_df.to_csv(collision_csv_filename, index=False)
else:
    # Create a DataFrame with all turbines and a Collision_Count of 0 if no collisions were found
    collision_counts = pd.DataFrame(turbine_coords, columns=['Turbine_Longitude', 'Turbine_Latitude'])
    collision_counts['Collision_Count'] = 0
    # Save the collision counts to a CSV file
    collision_counts.to_csv(collision_csv_filename, index=False)

print("Collision counts saved to 'wind_turbines_with_collisions.csv'.")

end_time = time.time()
print(f"Process completed in {end_time - start_time} seconds.")
