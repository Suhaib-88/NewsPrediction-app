import sqlite3
from Application_Logger import logger
import csv

class DBoperator:
    def __init__(self) -> None:
        self.log_writer=logger.App_Logger()
        self.file_object=open("All_logs/logs.txt",'a+')


    def Create_table(self):
        self.log_writer.log(self.file_object,"Connecting to `database1.db`")
        conn  =  sqlite3 . connect ( 'database1.db' ) #write database name
        cursor  =  conn.cursor ()
        try:
            cursor.execute("""CREATE TABLE predictions(predText text, predCategory text)""")
            self.log_writer.log(self.file_object,"Creating table with columns[predText,predCategory]")

        
        except Exception as e:
            pass
            self.log_writer.log(self.file_object,f"{e}")
            
        return conn,cursor


    def Insert_table(self,conn,cursor,texts,target_category):
        try:
            cursor.execute("""INSERT INTO predictions(predText,predCategory)
                VALUES (?,?)""", (str([texts]),str([target_category])))
            conn.commit ()
            self.log_writer.log(self.file_object,'Sucessfully entered data into TABLE predictions')
            data= cursor.execute("""SELECT * FROM predictions""")
        
            return conn,data

        except Exception as e:
            self.log_writer.log(self.file_object,f"Error occured at TableInsertion {e}")

        

    def fetch_to_csv(self,data,conn):
        try:
            with open('PredictionValidation/predictions_db', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(data)
                conn.close ()

        except Exception as e:
            self.log_writer.log(self.file_object,f"Error occured at Converting db into csv {e}")

        
