import cx_Oracle
import configparser

config = configparser.ConfigParser()	
config.optionxform = str
config.read("config.ini")

# files = ['initial_tables.sql','PAE_creation.sql' ]
files = [x.replace(" ", '') for x in config.get('initial_exexute', 'files').split(',')]

# Connect to the database
db_hostname=config.get('CONNECTION', 'hostname')
db_port=config.getint('CONNECTION', 'port')
db_sid=config.get('CONNECTION', 'sid')
db_user=config.get('CONNECTION', 'user')
db_password=config.get('CONNECTION', 'password')
db_encoding=config.get('CONNECTION', 'encoding')
dsn=cx_Oracle.makedsn(db_hostname,db_port,db_sid)

connection=cx_Oracle.connect(user=db_user, password=db_password, dsn=dsn, encoding=db_encoding)
cursor = connection.cursor()

for i in range(len(files)):
    f = open('initial_feed/'+files[i])
    full_sql = f.read()
    sql_commands = full_sql.split(';')

    for sql_command in sql_commands[:-1]:
        print(sql_command)
        s = sql_command+';'
        cursor.execute(sql_command)