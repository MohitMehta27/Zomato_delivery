import sys
sys.path.append("C:/Zomato delivery")
from dataclasses import dataclass 

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_transformation_object(self):
        try:
            logging.info("Data Transformation initiated")
            #seperating categorical numerical data 
            numericals_cols=['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition','multiple_deliveries', 'Time_taken (min)', 'Distance (in KM)']
            categorical_cols=['Weather_conditions', 'Road_traffic_density', 'Type_of_vehicle','Festival', 'City']


            #mapping according to rank
            weather_condition_map=["Sunny","Stormy","Sandstorms","Windy","Fog","Cloudy"]
            road_traffic_map=["Low","Medium","High","Jam"]
            type_of_vehicle_map=["electric_scooter", "scooter","bicycle","motorcycle"]
            festival_map=["No","Yes"]
            city_map=["Urban","Metropolitian", "Semi-Urban"]

            logging.info ("Pipeline initiated")
            ##Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer (strategy='median')),
                    ("scaler", StandardScaler())
                    ]
            )

            # Categorical pipeline
            cat_pipeline= Pipeline(
                steps=[
                    ('imputer', SimpleImputer (strategy="most_frequent")),
                    ('ordinalencoder',OrdinalEncoder(categories=[weather_condition_map,road_traffic_map,type_of_vehicle_map,festival_map,city_map])),
                    ('scaler', StandardScaler())
                    ]

            )
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numericals_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)


            ])
            
            return preprocessor

            logging.info('Pipeline Completed')


        except Exception as e:
            logging.info ("Error in data transformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
