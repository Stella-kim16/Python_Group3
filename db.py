# db.py
import os                                
from urllib.parse import quote_plus    
from dotenv import load_dotenv           
from sqlalchemy import create_engine     

load_dotenv()  # load .env file


user = os.getenv("DB_USER")                       # → "root"
password = quote_plus(os.getenv("DB_PASSWORD"))   # → pw
host = os.getenv("DB_HOST")                       # → "localhost"
port = os.getenv("DB_PORT")                       # → "3306"
db_name = os.getenv("DB_NAME")                    # → "energy_db"


engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}")

if __name__ == "__main__":
    with engine.connect() as conn:
        print("successfully connected to MySQL database:", db_name)