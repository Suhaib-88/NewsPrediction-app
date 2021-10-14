import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_selection 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report

import seaborn as sns
import matplotlib.pyplot as plt

classifier= MultinomialNB()
class ModelBuilding:
    """
    This class builds a machine learning model and evaluates for best perfomance.
    
    Written By: Suhaib
    Version: 1.0
    Revisions: 2.0
     """
    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_object= logger_object

    def train_test_splitting(self,df):
        """
        Method Name: train_test_splitting
        Description: Splits data into train and test set.
        Output: Numpy array of train and test sets.
        On Failure: Raise Exception

        Written By: Suhaib
        Version: 1.0
        Revisions: 1.0
        """
        try:
            df_train,df_test=train_test_split(df,test_size=0.25,random_state=np.random.seed(2),shuffle=True)
            return df_train,df_test

        except Exception as e:
            self.logger_object.log(self.file_object,f'Error occured while splitting the model {e}')
            raise Exception()

            
    def vectorize(self,df,df_train,df_test):
        """
        Method Name: vectorize
        Description: Fits a TfidfVectorizer on train set.
        Output: Returns a model fitted on train set
        On Failure: Raise Exception

        Written By: Suhaib
        Version: 1.0
        Revisions: 2.0
        """
        try:
            corpus=df_train.text.values
            vectorizer=TfidfVectorizer(max_features=3000,ngram_range=(1,2))
            vectorizer.fit(corpus)
            X_train= vectorizer.transform(corpus)
            y_train = df_train["category"].values
            dict_vocab=vectorizer.vocabulary_ 
            return X_train,y_train,vectorizer
        
        except Exception as e:
            self.logger_object.log(self.file_object,f'Error occured while Vectorizing train set {e}')
            raise Exception()

    def get_top_features(self,df,df_train,vectorizer,X_train):
        """
        Method Name: get_top_features
        Description: Extracts top significant words of our df_train.
        Output: Returns list containing most significant words to be used as vocabulary.
        On Failure: Raise Exception
        
        Written By: Suhaib
        Version: 1.0
        Revisions: 1.0
        """    
        try:
            y = df_train["category"]
            X_names = vectorizer.get_feature_names()
            p_value_limit = 0.95
            dtf_features = pd.DataFrame()
            for cat in np.unique(y):
                chi2, p = feature_selection.chi2(X_train, y==cat)
                dtf_features = dtf_features.append(pd.DataFrame(
                            {"feature":X_names, "score":1-p, "y":cat}))
                dtf_features = dtf_features.sort_values(["y","score"], 
                                ascending=[True,False])
                dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
            X_names = dtf_features["feature"].unique().tolist()
            return X_names

        except Exception as e:
            self.logger_object.log(self.file_object,f'Error occured while extracting top features {e}')
            raise Exception()
    
    def vectorize_vocab(self,X_names,df_train):
        """
        Method Name: vectorize_vocab
        Description: Vectorize model on a vocab of top significant words.
        Output: Returns a vectorized model based on significant vocab
        On Failure: Raise Exception 
        
        Written By: Suhaib
        Version: 1.0
        Revisions: 1.0
        """
        try:
            corpus=df_train.text.values
            vectorizer = TfidfVectorizer(vocabulary=X_names)
            vectorizer.fit(corpus)
            X_train = vectorizer.transform(corpus)
            vectorizer.vocabulary_
            return X_train,vectorizer   
       
        except Exception as e:
            self.logger_object.log(self.file_object,f'Error occured while Vectorizing top-feat vocabulary {e}')
            raise Exception()
    
    def model_fitting(self,X_train,y_train,df_test,vectorizer):
        """
         Method Name: model_fitting
         Description: Fits Vectorized Model on train set using PIPELINE.
         Output: Returns a fitted model, ready for predictions
         On Failure: Raise Exception
        
         Written By: Suhaib
         Version: 1.0
         Revisions: 2.0
        """
        try:
            model=Pipeline([("vectorizer",vectorizer),("classifier",classifier)])
            model['classifier'].fit(X_train,y_train)        
            return model
       
        except Exception as e:  
            self.logger_object.log(self.file_object,f'Error occured while fitting the model {e}')
            raise Exception()
    
    def model_prediction(self,df_test,model):
        """
         Method Name: model_prediction
         Description: Predicts model on test set.
         Output: Predicted Probability values for each category
         On Failure: Raise Exception    
        
         Written By: Suhaib
         Version: 1.0
         Revisions: None
        """
        try:
            x_val = df_test["text"].values
            y_pred=model.predict(x_val)
            pred_proba= model.predict_proba(x_val)
            y_val= df_test['category'].values
            return y_pred,y_val

        except Exception as e:
            self.logger_object.log(self.file_object,f'Error occured while predicting model {e}')
            raise Exception()

    def measure_metrics(self,y_pred,y_val):
        """
        Method Name: measure_metrics
        Description: Evaluates the model using accuracy_score and classif_report.
        Output: Plots heatmap of confusion matrix
        On Failure: Raise Exception
    
        Written By: Suhaib
        Version: 1.0
        Revisions: 1.0
        """
        try:
            sns.heatmap(confusion_matrix(y_pred,y_val),cmap='Reds')
            plt.show()
            print(accuracy_score(y_val,y_pred),classification_report(y_val,y_pred))

        except Exception as e:
            self.logger_object.log(self.file_object,f'Error occured while evaluating model performance {e}')
            raise Exception()

    def save_model(self,model):
        """
        Method Name: save_model
        Description: Saves Model using joblib.
        Output: A Saved model which can be loaded and deployed
        On Failure: Raise Exception
        
        Written By: Suhaib
        Version: 1.0
        Revisions: None
        """
        try:
            with open('Model.pkl', 'wb') as file:
                pickle.dump(model,file)
        
        except Exception as e:
            self.logger_object.log(self.file_object,f'Error occured while saving the model {e}')
            raise Exception()