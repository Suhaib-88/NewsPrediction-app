import pandas as pd

class DataGetter:
    """
    This class is used for retrieve the data from the source for training.

    Written By: Suhaib
    Version: 1.0
    Revisions: None

    """
    def __init__(self,path,file_object,logger_obj):
        self.input_file=path
        self.logger_obj=logger_obj
        self.file_object= file_object

    
    def fetch_data(self):
        """
            Method Name: fetch_data
            Description: This method reads the input files from source.
            Output: A pandas DataFrame.
            On Failure: Raise Exception

            Written By: Suhaib 
            Version: 1.0
            Revisions: None

        """

        self.logger_obj.log(self.file_object,'Entering data...')
        try:
            self.data=pd.read_csv(self.input_file)
            self.logger_obj.log(self.file_object,'Data is Successfully loaded')
            return self.data
        except FileNotFoundError as e:
            print(e)
            self.logger_obj.log(self.file_object,f"Error due to {e}")
            