import pandas as pd
import glob
import os

#Create combined_bird_data.csv from bird_data.csv files

# Go up one directory from the current working directory
parent_directory = os.path.dirname(os.getcwd())

# Define the pattern to search for 'bird_data.csv' files in the parent directory and its subdirectories
pattern = os.path.join(parent_directory, '**', 'bird_data.csv')

# Use glob to find all matching files with recursive search
bird_data_files = glob.glob(pattern, recursive=True)

# Initialize an empty DataFrame for the combined data
combined_data = pd.DataFrame(columns=['Latitude', 'Longitude', 'Timestamp', 'Species'])

# Process each found 'bird_data.csv' file
for file_path in bird_data_files:
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Remove rows missing any data in 'Latitude', 'Longitude', 'Timestamp', 'Species'
    data.dropna(subset=['Latitude', 'Longitude', 'Timestamp', 'Species'], inplace=True)
    
    # Append the cleaned data to the combined DataFrame
    combined_data = pd.concat([combined_data, data], ignore_index=True)

# Define the filename for the new combined CSV file
output_file = os.path.join(os.getcwd(), 'combined_bird_data.csv')  # Save in the current working directory

# Save the combined data to a new CSV file
combined_data.to_csv(output_file, index=False)

print(f"Combined data saved to {output_file}. Total files processed: {len(bird_data_files)}")
