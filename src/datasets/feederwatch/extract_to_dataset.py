import os
import pandas as pd

# Use the current directory where the script is running
directory = os.getcwd()

# New CSV file to store the combined data
output_file = 'bird_data.csv'

# Load the species translation table
translation_file = os.path.join(directory, 'PFW_spp_translation_table_May2023.csv')
translation_df = pd.read_csv(translation_file)
translation_df['species_code'] = translation_df['species_code'].str.lower()

# Initialize an empty DataFrame with desired column headers
combined_data = pd.DataFrame(columns=['Latitude', 'Longitude', 'Timestamp', 'Species'])

# Counter for the number of CSV files processed
csv_files_processed = 0

# Define the filename to exclude from processing
exclude_file = 'PFW_spp_translation_table_May2023.csv'

# Scan for each CSV file in the current directory, excluding the translation table
for filename in os.listdir(directory):
    if filename.endswith('.csv') and filename != exclude_file:
        file_path = os.path.join(directory, filename)
        
        # Open and read the CSV file
        try:
            data = pd.read_csv(file_path)
            print(f"Processing file: {filename}")
            
            # Prepare the data
            if 'SPECIES_CODE' in data.columns:
                data['species_code'] = data['SPECIES_CODE'].str.lower()
                data = pd.merge(data, translation_df[['species_code', 'american_english_name']], on='species_code', how='left')
                data['Species'] = data['american_english_name']
                data.drop(columns=['SPECIES_CODE', 'species_code', 'american_english_name'], inplace=True)
                
                # Assuming your data has columns named 'LATITUDE' and 'LONGITUDE'
                if 'LATITUDE' in data.columns and 'LONGITUDE' in data.columns:
                    data.rename(columns={'LATITUDE': 'Latitude', 'LONGITUDE': 'Longitude'}, inplace=True)
                
                # Merge date columns into 'Timestamp'
                if all(col in data.columns for col in ['Year', 'Month', 'Day']):
                    data['Timestamp'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
                    data['Timestamp'] = data['Timestamp'].dt.strftime('%m/%d/%Y')
                    data.drop(columns=['Year', 'Month', 'Day'], inplace=True)
                
                # Select only the desired columns
                data = data[['Latitude', 'Longitude', 'Timestamp', 'Species']]
                
                # Drop rows with any missing values in the required columns
                data.dropna(inplace=True)
                
                if not data.empty:
                    combined_data = pd.concat([combined_data, data], ignore_index=True)
                    csv_files_processed += 1
                else:
                    print(f"No rows left after dropping NAs in file: {filename}")
                
        except Exception as e:
            print(f"An error occurred with file {file_path}: {e}")

# Check if combined data is not empty before saving
if not combined_data.empty:
    combined_data.to_csv(os.path.join(directory, output_file), index=False)
    print(f"Combined data saved to {output_file} in the current directory.")
else:
    print("No data was combined; output CSV will not be created.")

print(f"Number of CSV files processed: {csv_files_processed}")
