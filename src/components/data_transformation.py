import os,sys
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler


from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np 
from dataclasses import dataclass
from src.utlis import save_object


@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated.')
            #Define which column should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            #Defining the custom ranking to each ordinal variables 

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline Initiated')

            ##Numerical Pipeline 

            num_pipeline =Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            ##Categorical_Pipeline 
            cat_pipeline =Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
                
            ])
            return preprocessor
            logging.info('Pipeline completed')

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)

    def initaite_data_transforation(self,train_data_path,test_data_path):
        try:
            #reading train and test data 

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Read train and test data completed')
            logging.info(f"Train Dataframe Head :\n{train_df.head().to_string()}")
            logging.info(f'Test Dataframe Head:\n{test_df.head().to_string()}')

            logging.info("Obtaining preprocessing Object")

            preprocessing_obj = self.get_data_transformation_object()

            target_columns ='price'
            drop_columns = [target_columns,'id']

            ##dividing dataset into dependent and independent dataset
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_columns]
            
            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_columns]
            
            
            #transformating usnig preprocessor obj 
            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and testing dataset')

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved.')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info('Exception Error Occured during Transformation')
            raise CustomException(e,sys)

