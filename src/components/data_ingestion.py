import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import geopy

from geopy.distance import geodesic

@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

## Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def calculate_distance(self, row):
        restaurant_coords = (row['Restaurant_latitude'], row['Restaurant_longitude'])
        delivery_coords = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
        return geodesic(restaurant_coords, delivery_coords).kilometers

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            df = pd.read_csv(os.path.join('notebook','data/finalTrain.csv'))
            logging.info('Dataset read as pandas Dataframe')

            # Calculate distance and create a new column
            df['Distance_in_km'] = df.apply(self.calculate_distance, axis=1)

            # Drop unnecessary columns
            df = df.drop(['ID',"Delivery_person_ID",'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude','Delivery_location_longitude'], axis=1)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=True)
            logging.info('Train test split')
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occurred at Data Ingestion stage')
            raise CustomException(e, sys)

''' wrote this code just to checkn if data_ingestion is working fine

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    print("Train Data Path:", train_data_path)
    print("Test Data Path:", test_data_path)
    '''

