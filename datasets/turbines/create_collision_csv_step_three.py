
import pandas as pd

# Load the wind turbine data
turbine_df = pd.read_csv('wind_turbine_location_20231128.csv')

# Create a new DataFrame with only longitude, latitude, and an initialized collision column
turbine_location_df = turbine_df[['xlong', 'ylat']].copy()
turbine_location_df['collision'] = 0

# Load the collision data
collision_df = pd.read_csv('detailed_wind_turbine_collisions_bk.csv')

# Rename the columns in the collision DataFrame for easier merging
collision_df_renamed = collision_df.rename(columns={"Turbine_Longitude": "xlong", "Turbine_Latitude": "ylat"})

# Aggregate the collision data to count collisions for each unique pair of longitude and latitude
collision_counts = collision_df_renamed.groupby(['xlong', 'ylat']).size().reset_index(name='collision_count')

# Merge the turbine location DataFrame with the aggregated collision counts on longitude and latitude
combined_df = pd.merge(turbine_location_df, collision_counts, on=['xlong', 'ylat'], how='left')

# Update the collision column with the collision counts, filling missing values with the original zeros
combined_df['collision'] = combined_df['collision_count'].fillna(0).astype(int)

# Drop the temporary collision count column
combined_df.drop(columns=['collision_count'], inplace=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('wind_turbines_with_collisions.csv', index=False)

print("CSV file has been saved to 'wind_turbines_with_collisions.csv'")
num_rows_combined = combined_df['collision'].sum()
print("Number of rows combined", num_rows_combined)