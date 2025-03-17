from enum import StrEnum


class TaskType(StrEnum):
    Train = "train"
    Valid = "valid"
    Test = "test"