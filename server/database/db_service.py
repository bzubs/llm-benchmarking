import bcrypt
import os
from dotenv import load_dotenv
from typing import Union

from database.database import DataBase
from schemas.user import User
from schemas.task import BenchTask, BenchResult

load_dotenv()



class DBService:
    def __init__(self, db: DataBase):
        self.db = db

    # ===================== USERS =====================

    def register_user(self, username: str, password: str): #access_code: str):
        
        # access_code_env = os.getenv("ACCESS_CODE")
        # if access_code != access_code_env:
        #     return False, "Incorrect Access Code"

        existing_user = self.db.get_user(username)
        if existing_user:
            return False, "Username already exists"

        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        user = User(username=username, password=hashed.decode())

        self.db.create_user(user)
        return True, "User registered"

    def login_user(self, username: str, password: str):
        user = self.db.get_user(username)

        if not user:
            return False, "Invalid credentials"

        stored_hash = user["password"].encode()

        if bcrypt.checkpw(password.encode(), stored_hash):
            return True, "Login successful"

        return False, "Invalid credentials"

    def update_user(self, username: str, update_data: dict):
        self.db.update_user(username, update_data)
        return True, "User updated"

    # ===================== TASKS =====================

    def create_task(
        self,
        task: BenchTask,
        backend_url: str,
        backend_task_id: str,
        logged: bool = False,
    ):
        task_dict = task.model_dump()
        task_dict.pop("id", None)

        task_dict.update(
            {
                "backend_url": backend_url,
                "backend_task_id": backend_task_id,
                "logged": logged,
            }
        )

        resp = self.db.create_task(task_dict)
        return True, str(resp.inserted_id), "Task created"

    def get_user_tasks(self, username: str):
        tasks = self.db.get_tasks_by_user(username)

        for task in tasks:
            task["_id"] = str(task["_id"])

        return tasks

    def get_task(self, task_id: str):
        task = self.db.get_task(task_id)
        if task:
            task["_id"] = str(task["_id"])
        return task


    def update_task_status(
    self,
    task_id: str,
    status: str,
    result,
    updated_at,
    ):
    # convert result → dict safely
        if result is not None and hasattr(result, "model_dump"):
            result = result.model_dump()

        res = self.db.update_task(
            task_id,
            {
                "status": status,
                "updated_at": updated_at,
                "result": result,
            },
        )

        if res is None:
            return False, "Invalid task_id"

        if res.matched_count == 0:
            return False, "Task not found"

        return True, "Task updated"

    def mark_task_logged(self, task_id: str):
        res = self.db.update_task(task_id, {"logged": True})

        if res is None:
            return False, "Invalid task_id"

        if res.matched_count == 0:
            return False, "Task not found"

        return True, "Task marked logged"
