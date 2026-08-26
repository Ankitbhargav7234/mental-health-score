import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from typing import Literal
#trained model
model = joblib.load('Mental_Health_Model.pkl')

app = FastAPI()
top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#Request body model
class StudentData(BaseModel):
    age                     : int = Field(..., ge=10, le=100)
    gender                  : Literal['Male', 'Female']
    country                 : str
    academic_level          : Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform      : Literal['Facebook','Instagram','Twitter','Snapchat','TikTok','YouTube','LinkedIn','LINE','WhatsApp','WeChat','KakaoTalk','VKontakte']
    purpose_of_use          : Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours   : float  = Field(..., ge=0, le=24)
    daily_unlocks           : int    = Field(..., ge=0)
    study_hours             : float = Field(..., ge=0, le=24)
    physical_activity_hours : float = Field(..., ge=0, le=24)
    sleep_hours_per_night   : float = Field(..., ge=0, le=24)
    stress_level            : Literal['Low', 'Medium', 'High', 'Very High']

#Response body model
class PredictionResponse(BaseModel):
    prediction_score: float

@app.get('/')
def home():
   return {"Welcome to the Mental Health Prediction APP!"} 

@app.post('/predict', response_model=PredictionResponse)
def predict(data: StudentData):
    country_group = 'Other' if data.country not in top_countries else data.country
    input_data = pd.DataFrame([
        {
            'Age'                     : data.age,
            'Gender'                  : data.gender,
            'Country'                 : data.country,
            'Academic_Level'          : data.academic_level,
            'Most_Used_Platform'      : data.most_used_platform,
            'Purpose_Of_Use'          : data.purpose_of_use,
            'Avg_Daily_Usage_Hours'   : data.avg_daily_usage_hours,
            'Daily_Unlocks'           : data.daily_unlocks,
            'Study_Hours'             : data.study_hours,
            'Physical_Activity_Hours' : data.physical_activity_hours,
            'Sleep_Hours_Per_Night'   : data.sleep_hours_per_night,
            'Stress_Level'            : data.stress_level,
            'Group_Country'           : country_group
        }
    ])

    prediction = model.predict(input_data)[0]
    return PredictionResponse(prediction_score=round(float(prediction), 2))