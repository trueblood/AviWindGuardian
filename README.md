# AviWind Guardian

Welcome to AviWind Guardian, a user-centric platform designed within the Dash framework to ensure a seamless experience for data science experts and beginners alike. By leveraging Python, our platform removes the complexity of web development, providing an accessible and powerful tool for ecological conservation through wind energy optimization.

## Features

- **Interactive Collision Prediction Dashboard**: A single-page application that intuitively combines simplicity with advanced functionality. Users can visualize wind turbine locations on an interactive Leaflet map and view potential collision events in a detailed results grid.

- **Machine Learning Model Feedback**: Integrates a direct feedback system allowing users to refine model predictions with 'thumbs up' or 'thumbs down' responses, enhancing the model's accuracy over time.

- **Dynamic Informative Labels**: Utilizes turbine data and wind insights from api.weather.gov to generate labels that assist in identifying optimal locations for turbine installations, effectively reducing the risk of avian collisions.

- **Forecasting Model with Interactive Controls**: Features sliders for adjusting the forecasting period, automatically updating predictions with the addition of new turbine data points.

- **Enhanced Navigation**: Includes 'Learn' and 'Training' tabs for easy navigation and model updates, ensuring a meaningful user interaction.

## Datasets

AviWind Guardian utilizes the following datasets:

- **Bird Data**: Leveraging comprehensive bird migration and sighting data from:
  - [USGS](https://www.usgs.gov/)
  - [Feederwatch](https://feederwatch.org/)
  - [Movebank: Bird Migration Data](https://datarepository.movebank.org/)

- **Wind Turbine Data**: Sourced from the [U.S. Geological Survey (USGS) Wind Turbine Database (WTDB)](https://eerscmap.usgs.gov/uswtdb/), providing detailed information on turbine locations across the United States.

- **Wind Speed Data**: Obtained from [api.weather.gov](https://api.weather.gov/), offering real-time wind data to enhance model predictions.

## Getting Started

To get started with AviWind Guardian, follow these steps:

```bash
# Clone the repository
git clone https://github.com/trueblood/AviWindGuardians.git

# Navigate to the project directory
cd AviWindGuardians

# Install dependencies
python scripts/install_dependencies.py

# Extract and normalize data for each dataset
# Replace <dataset_folder> with the actual path to your dataset folder
python <dataset_folder>/extract_to_dataset.py

# Navigate to the turbines folder to prepare bird and turbine data
cd turbines

# Combine bird data files
python create_bird_data_set_step_one.py

# Generate turbines with collision data (optimized for AMD GPUs)
python create_turbines_with_collision_gpu_step_two.py

# Create the final collision data CSV
python create_collision_csv_step_three.py

# Train the model
cd ..
python scripts/train_model.py

# Launch the application
cd app
python app.py
