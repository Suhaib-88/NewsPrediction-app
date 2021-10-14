import pandas as pd
from Application_Logger import logger 
from Data_Insertion import data_loader
from Data_Preprocessing import preprocess
from models import model_building
import spacy

class trainModel:
    
    def __init__(self):
        self.log_writer=logger.App_Logger()
        self.file_object=open("All_logs/logs.txt",'a+')
    
    
    def preprocessing_data(self):
        self.log_writer.log(self.file_object,"Start of Training")
        try:
            self.log_writer.log(self.file_object,'Fetching raw data')
            data_getter= data_loader.DataGetter("Input_data/input_news_files.csv",self.file_object,self.log_writer)
            data= data_getter.fetch_data()
            
            """Performing data preprocessing now"""
            df_preprocessed=preprocess.PreProcessing(self.file_object,self.log_writer)
            data= df_preprocessed.merge_and_drop_cols(data)
            data=df_preprocessed.remove_nans(data)
            data=df_preprocessed.equalize(data)
            data= df_preprocessed.clean_texts(data)
            self.log_writer.log(self.file_object,'preprocessing done')
            return data.to_csv('Submission_data/Output_news_files.csv',index=False)
            
        except Exception as e:
            self.log_writer.log(self.file_object,f'An error occured {e}')   


    def training_model(self):
        self.log_writer.log(self.file_object,'Fetching processed data file')
        try:
            data_getter= data_loader.DataGetter("Submission_data/Output_news_files.csv",self.file_object,self.log_writer)
            data= data_getter.fetch_data()
            data=data.dropna()
            data=data.reset_index()

            
            """Performing model building with train set"""
            df_trainer= model_building.ModelBuilding(self.file_object,self.log_writer)
            df_train,df_test= df_trainer.train_test_splitting(data)
            self.log_writer.log(self.file_object,'Splitting model into train and test set')
            X_train,y_train,vectorizer=df_trainer.vectorize(data,df_train,df_test)
            self.log_writer.log(self.file_object,'Vectorizing the model with TFIDF')
            X_names=df_trainer.get_top_features(data,df_train,vectorizer,X_train)
            self.log_writer.log(self.file_object,'Extract the top features in texts')
            X_train, vectorizer= df_trainer.vectorize_vocab(X_names,df_train)
            self.log_writer.log(self.file_object,'Vectorizing the on our top vocabulary')
            model=df_trainer.model_fitting(X_train,y_train,df_test,vectorizer)
            self.log_writer.log(self.file_object,'Model fitted Done')
            
            """Performing predictions on test set"""
            y_pred,y_val=df_trainer.model_prediction(df_test,model)
            self.log_writer.log(self.file_object,'Predictions are made')
            
            """Evaluating performance"""
            df_trainer.measure_metrics(y_pred,y_val)
            self.log_writer.log(self.file_object,'Metrics to evaluate performance')
            
            """Saving Model"""
            df_trainer.save_model(model)
            self.log_writer.log(self.file_object,'Saved Model Successfully')
        
        except Exception as e:
            self.log_writer.log(self.file_object,f'An error occured {e}')



if __name__=="__main__":
    model=trainModel()
    # model.training_model()   # comment this line to preprocess data first
    model.preprocessing_data() # then comment this line to train data
