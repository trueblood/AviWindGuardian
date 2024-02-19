import pandas as pd

#this is the code to create a new csv file with the timestamp of the collision
#for forecasting model

# Load wind turbine location data
wind_turbine_data = pd.read_csv('wind_turbine_location_20231128.csv')

# Load combined bird data
bird_data = pd.read_csv('combined_bird_data.csv')

# Assume 'combined_bird_data.csv' has columns named 'longitude', 'latitude', and 'timestamp'
# Adjust column names as necessary based on the actual structure of 'combined_bird_data.csv'

matches = []
for index, row in wind_turbine_data.iterrows():
    # Find matching rows in bird data based on longitude and latitude
    matched_rows = bird_data[
        (bird_data['longitude'] == row['xlong']) & 
        (bird_data['latitude'] == row['ylat'])
    ]
    
    # Group by timestamp to count collisions at each timestamp
    collision_counts = matched_rows.groupby('timestamp').size().reset_index(name='collisions')
    
    # If matches are found, append their details along with the collision count to the list
    for _, match in collision_counts.iterrows():
        matches.append({
            'xlong': row['xlong'],
            'ylat': row['ylat'],
            'timestamp': match['timestamp'],
            'total_collisions': match['collisions']
        })

# Convert matches to a DataFrame
matched_data = pd.DataFrame(matches)

# Optionally, save the matched data to a CSV file
matched_data.to_csv('matched_bird_turbine_data.csv', index=False)
