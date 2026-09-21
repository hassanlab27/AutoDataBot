import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_clean_csv_bytes() -> bytes:
    csv_text = (
        "age,gender,income,target\n"
        "25,Female,50000,0\n"
        "30,Male,60000,1\n"
        "35,Female,75000,1\n"
        "40,Male,80000,0\n"
        "45,Female,95000,1\n"
    )
    return csv_text.encode("utf-8")

@pytest.fixture
def sample_messy_csv_bytes() -> bytes:
    # Contains: duplicate row, duplicate column name, unnamed index, missing values, constant column
    csv_text = (
        "Unnamed: 0,user_id,age,age,status,constant_col,notes\n"
        "0,usr_101,25,25,active,ONE,hello world\n"
        "1,usr_102,30,,inactive,ONE,test note\n"
        "2,usr_103,,35,active,ONE,\n"
        "3,usr_101,25,25,active,ONE,hello world\n"  # duplicate row
    )
    return csv_text.encode("utf-8")

@pytest.fixture
def sample_semicolon_csv_bytes() -> bytes:
    csv_text = (
        "city;temperature;rainfall\n"
        "Paris;18.5;2.1\n"
        "London;15.2;5.4\n"
        "Tokyo;22.0;0.0\n"
    )
    return csv_text.encode("latin-1")
