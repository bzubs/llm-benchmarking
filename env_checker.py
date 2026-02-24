from dotenv import load_dotenv
import os

load_dotenv()  
access_code = os.getenv("ACCESS_CODE")

print(access_code)