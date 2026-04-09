from pymongo import MongoClient
from bson import ObjectId
from bson.errors import InvalidId

from schemas.user import User
from schemas.task import BenchTask


class DataBase:
    def __init__(self, url: str):
        self.url = url
        self.client = None
        self.db = None

    def _get_db(self):
        if self.db is None:
            raise Exception("Database not connected. Call connect() first.")
        return self.db

    def connect(self, db_name: str) -> MongoClient | None:
        try:
            self.client = MongoClient(self.url)
            self.db = self.client[db_name]
            return self.client
        except Exception as e:
            print(f"Connection failed: {e}")
            return None

    # ===================== USERS =====================

    def create_user(self, user: User):
        self.db = self._get_db()
        return self.db["users"].insert_one(user.model_dump())

    def get_user(self, username: str):
        self.db = self._get_db()
        return self.db["users"].find_one({"username": username})

    def update_user(self, username: str, update_data: dict):
        self.db = self._get_db()
        return self.db["users"].update_one(
            {"username": username},
            {"$set": update_data},
        )

    # ===================== TASKS =====================

    def create_task(self, task_data: dict):
        self.db = self._get_db()
        return self.db["tasks"].insert_one(task_data)

    def get_task(self, task_id: str):
        self.db = self._get_db()
        try:
            obj_id = ObjectId(task_id)
        except InvalidId:
            return None
        return self.db["tasks"].find_one({"_id": obj_id})

    def get_tasks_by_user(self, username: str):
        self.db = self._get_db()
        return list(self.db["tasks"].find({"username": username}))

    def update_task(self, task_id: str, update_data: dict):
        self.db = self._get_db()
        try:
            obj_id = ObjectId(task_id)
        except InvalidId:
            return None
        return self.db["tasks"].update_one(
            {"_id": obj_id},
            {"$set": update_data},
        )
