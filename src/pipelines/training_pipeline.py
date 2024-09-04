import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd 

from src.components.data_ingestion import Dataingestion
from src.components.data_transformation import DataTransformation

if __name__== '__main__':
    obj=Dataingestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)
    data_transformation = DataTransformation()
    train_arr,test_arr,obj= data_transformation.initaite_data_transforation(train_data_path,test_data_path)
    
