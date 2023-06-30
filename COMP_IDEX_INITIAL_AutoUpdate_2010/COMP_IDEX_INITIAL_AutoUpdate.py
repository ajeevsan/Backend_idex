import cx_Oracle
import configparser

from datetime import datetime

from threading import Timer
import logging

config = configparser.ConfigParser()
config.optionxform = str
config.read("config.ini")

db_hostname = config.get('CONNECTION', 'hostname')
db_port = config.getint('CONNECTION', 'port')
# 14-03-2022 Manish , changed according to my system connection
db_sid = config.get('CONNECTION', 'sid')
db_user = config.get('CONNECTION', 'user')
db_password = config.get('CONNECTION', 'password')
db_encoding = config.get('CONNECTION', 'encoding')
dsn = cx_Oracle.makedsn(db_hostname, db_port, db_sid)

connection = cx_Oracle.connect(
    user=db_user, password=db_password, dsn=dsn, encoding=db_encoding)


# class definition of repeated timer
class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self._run()

    def _run(self):
        self.is_running = False
        self.function(*self.args, **self.kwargs)
        self.start()

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = True

# Function to insert Dummy records to Input table automatically
def put_timely_data():
    

    try:        
        connection = cx_Oracle.connect(
            user=db_user, password=db_password, dsn=dsn, encoding=db_encoding)
        cursor = connection.cursor()

        _oraQuery = "INSERT INTO COMP_IDEX4 SELECT * FROM COMP_IDEX4_2018_2022 "\
                    "WHERE TO_CHAR(TO_DATE((year || '-' || month || '-' || dd || ' ' || GGGG ||'00'), "\
                    "'YYYY-MM-DD HH24MISS'),  'YYYY-MM-DD HH24:MI:SS') = "\
                    "(SELECT MIN(DATETIME) AS LATEST_INITIAL_REC FROM( "\
                    "SELECT TO_CHAR(TO_DATE((YEAR || '-' || MONTH || '-' || DD || ' ' || "\
                    "GGGG), 'YYYY-MM-DD HH24MISS'), 'YYYY-MM-DD HH24:MI:SS') AS DATETIME "\
                    "FROM COMP_IDEX4_2018_2022 WHERE TO_CHAR(TO_DATE((YEAR || '-' || MONTH || '-' || DD || ' ' || "\
                    "GGGG), 'YYYY-MM-DD HH24MISS'), 'YYYY-MM-DD HH24:MI:SS') > "\
                    "(SELECT MAX(MAXDATETIME) AS LAST_COMPIDEX FROM( "\
                    "SELECT TO_CHAR(TO_DATE((YEAR || '-' || MONTH || '-' || DD || ' ' || "\
                    "GGGG), 'YYYY-MM-DD HH24MISS'), 'YYYY-MM-DD HH24:MI:SS') AS MAXDATETIME "\
                    "FROM COMP_IDEX4))))"
        cursor.execute(_oraQuery)
        connection.commit()

        print("____________________________________________________________")
        print()
        print(_oraQuery)        
        print()
        print(datetime.now(), "Dummy records insertion successful.")
        print("____________________________________________________________")
        print()

    except Exception as ex:
        print("Error occured in function put_timely_data ", ex)
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        



if __name__ == "__main__":
    try:
        rt1 = RepeatedTimer(20, put_timely_data)
    except Exception as ex:
        print("Error occured in function main ", ex)
        print("Complete Error =  ", str(logging.traceback.format_exc()))    
