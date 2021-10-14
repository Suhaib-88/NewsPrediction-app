import pandas as pd
from sklearn.utils import shuffle
from nltk.corpus import  stopwords
from nltk.stem import WordNetLemmatizer
import re


class PreProcessing:
    """
        This class performs cleaning and transformation of data prior to training.

        Written By: Suhaib
        Version: 1.0
        Revisions: None

    """

    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_object=logger_object
        
    
    def merge_and_drop_cols(self,df):
        """
        Method Name: merge_and_drop_cols
        Description: This method merges two columns and then  removes the given columns from a pandas dataframe.
        Output: A pandas DataFrame after removing the specified columns.
        On Failure: Raise Exception

        Written By: Suhaib
        Version: 1.0
        Revisions: None

        """
        try:
            df['text']= df['headline']+" "+df['short_description'] +' '+ df["keywords"]
            self.logger_object.log(self.file_object,'Merged columns: short_description & headline ')
            df.drop(["headline","short_description","keywords",'links'],axis=1,inplace=True)
            self.logger_object.log(self.file_object,'Dropped columns: short_description & headline ')
            return df
        
        except Exception as e:
            self.logger_object.log(self.file_object,f'Error occured while merging and dropping cols {e}')
            raise Exception()

    def remove_nans(self,df):
        """
        Method Name: remove_nans
        Description: This method Removes all missing values in the Dataframe.
        Output: Returns Dataframe with imputed missing values.
        On Failure: Raise Exception

        Written By: Suhaib
        Version: 1.0
        Revisions: 1.0
        """
        try:
            df.dropna(how='all',subset=['text'],inplace=True)
            df=df.reset_index(drop=True)
            self.logger_object.log(self.file_object,'removed nans')
            return df

        except Exception as e:
            self.logger_object.log(self.file_object,f'Error occured while imputing missing values {e}')
            raise Exception()

    def equalize(self,df1):
        """
        Method Name: equalize
        Description: This method handles the imbalance in target columns, by taking equal sample of each category.
        Output: new balanced dataframe and balanced target columns
        On Failure: Raise Exception
        Written By: Suhaib
        Version: 1.0
        Revisions: None
        """
        try:

            df1=shuffle(df1).reset_index(drop=True)
            self.logger_object.log(self.file_object,'balanced targets')
            return df1
            
        except Exception as e:
            self.logger_object.log(self.file_object,f'error occured during Balancing{e}')
            raise Exception()


    def clean_texts(self,df):
        """
        Method Name: clean_texts
        Description: This method cleans texts with re, applies lemmatizer and stopwords .
        Output: A pandas DataFrame with all the texts lemmatized and stopwords removed.
        On Failure: Raise Exception

        Written By: Suhaib
        Version: 1.0
        Revisions: 1.0

        """
        try:
            lemmatizer=WordNetLemmatizer()
            corpus=[]
            for i in range(0,len(df.text)):
                review=re.sub('[^a-zA-Z]',' ',str(df.text[i]))
                review=review.lower().split()
                    
                review=[lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
                review=' '.join(review)
                corpus.append(review)
            df['text']= corpus
            self.logger_object.log(self.file_object,'cleaned text data')
            return df

        except Exception as e:
            self.logger_object.log(self.file_object,f'Error occured while cleaning and lemmatizing {e}')
            raise Exception()
