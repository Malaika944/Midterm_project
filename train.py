import os
from dotenv import load_dotenv
from pymongo import MongoClient
from noshow_iq.preprocess import load_and_clean, get_features_and_target
from noshow_iq.model import train

load_dotenv()

DATASET_PATH = "data/raw/KaggleV2-May-2016.csv"

print("Loading dataset...")
df = load_and_clean(DATASET_PATH)
print(f"Rows after cleaning: {len(df)}")

X, y = get_features_and_target(df)
print(f"No-show rate: {y.mean():.1%}")

print("Connecting to MongoDB Atlas...")
client = MongoClient(os.getenv("MONGO_URI"))
db = client["noshowiq"]

print("Training model - this takes 1-2 minutes...")
model, X_test, y_test = train(X, y, db=db)
print("All done!")