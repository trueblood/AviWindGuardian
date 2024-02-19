import pandas as pd

class CollisionTracker:
    def __init__(self, turbine_file, migration_file):
        self.turbine_file = turbine_file
        self.migration_file = migration_file
        self.turbines_df = None
        self.migration_df = None
        self.collisions = []

    def load_data(self):
        self.turbines_df = pd.read_csv(self.turbine_file)
        self.migration_df = pd.read_csv(self.migration_file)

    @staticmethod
    def is_close(lon1, lat1, lon2, lat2, threshold=0.01):  # threshold in degrees, ~1km
        return abs(lon1 - lon2) < threshold and abs(lat1 - lat2) < threshold

    def find_collisions(self):
        for _, bird_row in self.migration_df.iterrows():
            bird_lon, bird_lat = bird_row['BirdLongitude'], bird_row['BirdLatitude']
            date = bird_row['Date']

            for _, turbine_row in self.turbines_df.iterrows():
                turbine_id = turbine_row['TurbineID']
                turbine_lon, turbine_lat = turbine_row['Longitude'], turbine_row['Latitude']

                if self.is_close(bird_lon, bird_lat, turbine_lon, turbine_lat):
                    self.collisions.append({'Date': date, 'TurbineID': turbine_id, 'Longitude': turbine_lon, 'Latitude': turbine_lat})

    def save_collisions(self, output_file='bird_turbine_collisions.csv'):
        collisions_df = pd.DataFrame(self.collisions)
        collisions_df.to_csv(output_file, index=False)
        print(collisions_df.head())  # Print the first few rows to check

if __name__ == '__main__':
    tracker = CollisionTracker('turbine_locations.csv', 'bird_migration.csv')
    tracker.load_data()
    tracker.find_collisions()
    tracker.save_collisions()
