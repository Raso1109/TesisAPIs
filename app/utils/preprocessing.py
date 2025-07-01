import numpy as np
from PIL import Image
import io

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))
    return np.array(image) / 255.0

def preprocess_features(features_dict):
    feature_order = [
        'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
        'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
        'FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'Hypertension',
        'Diabetes', 'Depression', 'Stroke', 'SystolicBP', 'DiastolicBP',
        'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
        'CholesterolTriglycerides', 'UPDRS', 'MoCA', 'FunctionalAssessment',
        'Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability',
        'SpeechProblems', 'SleepDisorders', 'Constipation'
    ]
    return np.array([float(features_dict[f]) for f in feature_order])
