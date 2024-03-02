import pandas as pd
import numpy as np
import pyopencl as cl
import os
import time
import pyamdgpuinfo

# This script is meant to run on AMD GPUs. If you have an NVIDIA GPU, you can use the CUDA Toolkit and PyCUDA to run similar code on your GPU. 

# PyOpenCL Setup
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)
start_time = time.time()

for platform in cl.get_platforms():
    for device in platform.get_devices():
        if device.type == cl.device_type.GPU:
            context = cl.Context([device])
            queue = cl.CommandQueue(context)
            print(f"Using GPU: {device.name}")
            break
    else:
        continue
    break
else:
    print("No GPU device found.")
    exit(1)


# Haversine Kernel in OpenCL
haversine_kernel = """
#define R 3956
#define TO_RAD (M_PI / 180.0)
__kernel void haversine(
    __global const float *lon1, __global const float *lat1,
    __global const float *lon2, __global const float *lat2,
    __global float *distances) {
    int i = get_global_id(0);
    float dlon = (lon2[i] - lon1[i]) * TO_RAD;
    float dlat = (lat2[i] - lat1[i]) * TO_RAD;
    float a = sin(dlat / 2) * sin(dlat / 2) + cos(lat1[i] * TO_RAD) * cos(lat2[i] * TO_RAD) * sin(dlon / 2) * sin(dlon / 2);
    float c = 2 * atan2(sqrt(a), sqrt(1 - a));
    distances[i] = R * c;
}
"""

program = cl.Program(context, haversine_kernel).build()


def print_memory_usage_gpu():
    first_gpu = pyamdgpuinfo.get_gpu(0) # returns a GPUInfo object
    vram_usage = first_gpu.query_vram_usage()
    vram_usage_in_gb = vram_usage / (1024 ** 3)
    gpu_temp_fahrenheit = first_gpu.query_temperature() * 9/5 + 32
    gpu_load = first_gpu.query_load()
    gpu_power = first_gpu.query_power()
    print(f"Current GPU memory usage: {vram_usage_in_gb} GB")
    print(f"Current GPU Load: {gpu_load}")
    print(f"Current GPU Power: {gpu_power} W")
    print(f"Current GPU Temp: {gpu_temp_fahrenheit} F")

# Batch processing function
def process_batch(turbine_coords, bird_coord, batch_size):
    num_turbines = turbine_coords.shape[0]
    print(f"Processing {num_turbines} turbines in batches of {batch_size}...")
    results = np.empty(num_turbines, dtype=np.float32)

    print("GPU metrics:")
    print_memory_usage_gpu()  # Print GPU metrics before processing

    for start in range(0, num_turbines, batch_size):
        end = min(start + batch_size, num_turbines)
        current_batch_size = end - start

        # Create buffers
        mf = cl.mem_flags
        turbine_lon_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=turbine_coords[start:end, 0])
        turbine_lat_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=turbine_coords[start:end, 1])
        bird_lon_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.full(current_batch_size, bird_coord[0], dtype=np.float32))
        bird_lat_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.full(current_batch_size, bird_coord[1], dtype=np.float32))
        distance_buf = cl.Buffer(context, mf.WRITE_ONLY, results[start:end].nbytes)
        # Execute the kernel
        program.haversine(queue, (current_batch_size,), None, turbine_lon_buf, turbine_lat_buf, bird_lon_buf, bird_lat_buf, distance_buf)

        # Read back the results
        cl.enqueue_copy(queue, results[start:end], distance_buf)
    return results

def get_bird_species_by_index(index, bird_data_df):
    """
    Fetches the species of the bird based on its index in a pandas DataFrame.
    
    :param index: The index of the bird in the bird_data_df DataFrame.
    :param bird_data_df: pandas DataFrame containing bird data.
    :return: The species of the bird at the given index.
    """
    if 0 <= index < len(bird_data_df):
        return bird_data_df.iloc[index]['Species']
    else:
        return None  # Or raise an error, depending on your preference

def get_bird_timestamp_by_index(index, bird_data_df):
    """
    Fetches the timestamp of the bird based on its index in a pandas DataFrame.
    
    :param index: The index of the bird in the bird_data_df DataFrame.
    :param bird_data_df: pandas DataFrame containing bird data.
    :return: The timestamp of the bird at the given index.
    """
    if 0 <= index < len(bird_data_df):
        return bird_data_df.iloc[index]['Timestamp']
    else:
        return None  # Or raise an error, depending on your preference
    
# Define function to load the last processed bird index
def load_last_processed_bird_index():
    if os.path.exists('last_processed_bird_index.txt'):
        with open('last_processed_bird_index.txt', 'r') as file:
            return int(file.read())
    else:
        return 0  # Start from the beginning if the file doesn't exist

# Define function to save the last processed bird index
def save_last_processed_bird_index(index):
    with open('last_processed_bird_index.txt', 'w') as file:
        file.write(str(index))
    
# Load Data
wind_turbines_df = pd.read_csv('wind_turbine_location_20231128.csv')
bird_data_df = pd.read_csv('combined_bird_data.csv')

# Preprocessing
wind_turbines_df.rename(columns={'xlong': 'Longitude', 'ylat': 'Latitude'}, inplace=True)
bird_data_list = bird_data_df[['Longitude', 'Latitude', 'Species', 'Timestamp']].to_dict('records')

# Convert data to numpy arrays for PyOpenCL
turbine_coords = wind_turbines_df[['Longitude', 'Latitude']].values.astype(np.float32)
bird_coords = bird_data_df[['Longitude', 'Latitude']].values.astype(np.float32)

#print('bird cords:', bird_coords)
#print('turbine cords:', turbine_coords)

last_processed_bird_index = load_last_processed_bird_index()
collisions_csv_filename = 'detailed_wind_turbine_collisions.csv'

# Example of processing collisions for each bird
collision_threshold = 1.0  # Collision threshold in miles
collisions = []

total_birds = len(bird_coords)

for bird_index in range(last_processed_bird_index, len(bird_coords)):
    bird_coord = bird_coords[bird_index]
    distances = process_batch(turbine_coords, bird_coord, batch_size=1024)
    collision_indices = np.where(distances <= collision_threshold)[0]
    
    for idx in collision_indices:
        turbine_longitude, turbine_latitude = turbine_coords[idx]
        bird_species = get_bird_species_by_index(bird_index, bird_data_df)
        bird_timestamp = get_bird_timestamp_by_index(bird_index, bird_data_df)
        
        # Append collision record to the CSV file
        with open(collisions_csv_filename, 'a') as file:
            file.write(f"{bird_species},{turbine_longitude},{turbine_latitude},{bird_timestamp},{distances[idx]}\n")
    
    # Save the current bird index for resuming after a crash
    save_last_processed_bird_index(bird_index + 1)
    
    # Calculate and print progress percentage
    progress_percentage = (bird_index + 1) / total_birds * 100
    print(f"Progress: {progress_percentage:.2f}%")

print("All collisions processed and saved.")

# collision_csv_filename = 'wind_turbines_with_collisions.csv'

# # Calculate collisions per turbine location, handling the case where no collisions are found
# if not collisions_df.empty:
#     print("collisions_df is not empty")
#     collision_counts = collisions_df.groupby(['Turbine_Longitude', 'Turbine_Latitude']).size().reset_index(name='Collision_Count')
#     # Merge this summary back into the original wind_turbines_df
#     # Use left merge to ensure all turbines are included, even those without collisions
#     wind_turbines_df = wind_turbines_df.merge(collision_counts, how='left', left_on=['Longitude', 'Latitude'], right_on=['Turbine_Longitude', 'Turbine_Latitude'])

#     # Fill NaN values in Collision_Count with 0 to indicate no collisions
#     wind_turbines_df['Collision_Count'] = wind_turbines_df['Collision_Count'].fillna(0)

#     # Keep only the required columns
#     wind_turbines_df = wind_turbines_df[['Longitude', 'Latitude', 'Collision_Count']]

#     # Save the enriched wind_turbines_df with collision counts
#     wind_turbines_df.to_csv(collision_csv_filename, index=False)
# else:
#     # Create a DataFrame with all turbines and a Collision_Count of 0 if no collisions were found
#     collision_counts = pd.DataFrame(turbine_coords, columns=['Turbine_Longitude', 'Turbine_Latitude'])
#     collision_counts['Collision_Count'] = 0
#     # Save the collision counts to a CSV file
#     collision_counts.to_csv(collision_csv_filename, index=False)

# print("Collision counts saved to 'wind_turbines_with_collisions.csv'.")

end_time = time.time()
print(f"Process completed in {end_time - start_time} seconds.")