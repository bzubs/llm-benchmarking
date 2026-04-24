from pydantic import BaseModel


class User(BaseModel):
    username : str
    password : str

class UserRegisterModel(BaseModel):
    username : str
    password : str
    access_code : str



