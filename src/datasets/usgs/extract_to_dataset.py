import os
import pandas as pd

# Use the current directory where the script is running
directory = os.getcwd()

# Column headers we're looking for and their new names
column_map = {
    'animal': 'Species',
    'time': 'Timestamp',
    'lat': 'Latitude',
    'long': 'Longitude'
}

# New CSV file to store the combined data
output_file = 'bird_data.csv'  # The file will be saved in the current directory

# Initialize an empty DataFrame with desired column headers
combined_data = pd.DataFrame(columns=column_map.values())

# Counter for the number of CSV files processed
csv_files_processed = 0

# Scan for each CSV file in the current directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        
        # Open and read the CSV file
        try:
            data = pd.read_csv(file_path)
            print(f"Processing file: {filename}")
            
            # Create a dictionary to map data columns to the required columns based on partial match
            column_mapping = {}
            for data_col in data.columns:
                for col_key in column_map.keys():
                    if col_key in data_col.lower():
                        column_mapping[data_col] = column_map[col_key]
                        break
            
            # If no required columns are present, skip this file
            if not column_mapping:
                print(f"No matching columns found in file: {filename}")
                continue
            
            # Select and rename the columns based on the mapping
            data = data[column_mapping.keys()].rename(columns=column_mapping)
            
            # Only keep rows where all required new columns have non-empty values
            data.dropna(subset=column_map.values(), inplace=True)
            
            # Check if after dropping NAs we still have data to append
            if not data.empty:
                # Append the filtered data to the combined DataFrame
                combined_data = pd.concat([combined_data, data], ignore_index=True)
                csv_files_processed += 1
            else:
                print(f"No rows left after dropping NAs in file: {filename}")
                
        except Exception as e:
            print(f"An error occurred with file {file_path}: {e}")

# Check if combined data is not empty before saving
if not combined_data.empty:
    # Save the combined data to a new CSV file in the current directory
    combined_data.to_csv(os.path.join(directory, output_file), index=False)
    print(f"Combined data saved to {output_file} in the current directory.")
else:
    print("No data was combined; output CSV will not be created.")

# Print the number of CSV files processed
print(f"Number of CSV files processed: {csv_files_processed}")
