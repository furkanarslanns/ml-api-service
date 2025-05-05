from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle

app = FastAPI()

# Model bileşenlerini yüklee
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("gender_encoder.pkl", "rb") as f:
    gender_encoder = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Mapping sözlükleri
mappings = {
    "Physical Activity Level": {
        "Sedentary": 0, "Lightly Active": 1, "Moderately Active": 2,
        "Very Active": 3, "Super Active": 4
    },
    "Daily Activity Level": {
        "İyi": 2, "Orta": 1, "Kötü": 0,
        "Poor": 0, "Moderate": 1, "Good": 2
    },
    "Prior Exercise Experience": {
        "Bir yıldan az": 0, "Bir yıl": 1, "Bir yıldan fazla": 2,
        "Less than a year": 0, "One year": 1, "More than a year": 2
    },
    "Sleep Quality": {
        "Excellent": 3, "Good": 2, "Fair": 1, "Poor": 0
    },
    "Goal": {
        "Kilo vermek": 0, "Kilo almak": 1, "Kas yapmak": 2,
        "Lose weight": 0, "Gain weight": 1, "Build muscle": 2
    }
}

# Kullanıcıdan alınacak veri modeli
class UserInput(BaseModel):
    age: int = Field(..., alias="age")
    gender: str = Field(..., alias="gender")
    currentWeightLbs: float = Field(..., alias="currentWeightLbs")
    weightChangeLbs: float = Field(..., alias="weightChangeLbs")
    durationWeeks: int = Field(..., alias="durationWeeks")
    physicalActivityLevel: str = Field(..., alias="physicalActivityLevel")
    sleepQuality: str = Field(..., alias="sleepQuality")
    stressLevel: int = Field(..., alias="stressLevel")
    goal: str = Field(..., alias="goal")
    priorExerciseExperience: str = Field(..., alias="priorExerciseExperience")
    dailyActivityLevel: str = Field(..., alias="dailyActivityLevel")
    height: float = Field(..., alias="height")  # meters

    class Config:
        allow_population_by_field_name = True

# Tahmin endpoint'i
@app.post("/predict")
def predict(user: UserInput):
    try:
        # Veriyi DataFrame'e çevir
        df = pd.DataFrame([{
            "Age": user.age,
            "Gender": user.gender,
            "Current Weight (lbs)": user.currentWeightLbs,
            "Weight Change (lbs)": user.weightChangeLbs,
            "Duration (weeks)": user.durationWeeks,
            "Physical Activity Level": user.physicalActivityLevel,
            "Sleep Quality": user.sleepQuality,
            "Stress Level": user.stressLevel,
            "Goal": user.goal,
            "Prior Exercise Experience": user.priorExerciseExperience,
            "Daily Activity Level": user.dailyActivityLevel,
            "Height (m)": user.height
        }])

        # Hesaplamalar
        df["Final Weight (lbs)"] = df["Current Weight (lbs)"] + df["Weight Change (lbs)"]
        df["Daily Caloric Surplus/Deficit"] = (
            df["Weight Change (lbs)"] * 3500
        ) / (df["Duration (weeks)"] * 7)

        # BMI hesapla 
        df["BMI"] = (df["Current Weight (lbs)"] * 0.453592) / (df["Height (m)"] ** 2)

        # BMR hesapla 
        gender_value = df["Gender"].iloc[0]
        if gender_value.lower() == "male":
            bmr = 10 * (df["Current Weight (lbs)"] * 0.453592) + 6.25 * (df["Height (m)"] * 100) - 5 * df["Age"] + 5
        else:
            bmr = 10 * (df["Current Weight (lbs)"] * 0.453592) + 6.25 * (df["Height (m)"] * 100) - 5 * df["Age"] - 161
        df["BMR (Calories)"] = bmr

        # Encoding ve mapping işlemleri
        df["Gender"] = gender_encoder.transform([df["Gender"].iloc[0]])
        for column, mapping in mappings.items():
            if column in df.columns:
                df[column] = df[column].map(mapping)

        # NaN temizliği ve sıralama
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.mode().iloc[0], inplace=True)
        df = df[model_columns]

        # Ölçekleme ve tahmin
        scaled = scaler.transform(df)
        pred = model.predict(scaled)[0]

        return {
         "dailyWaterIntake": float(round(pred[0], 2)),
    "dailyExerciseMin": float(round(pred[1])),  
    "dailyCaloriesConsumed": float(round(pred[2], 2))
        }

    except Exception as e:
         import traceback
         print("❌ Hata detayları:")
         traceback.print_exc()

         raise HTTPException(status_code=500, detail=str(e))

