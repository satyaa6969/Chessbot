import os

print("Current working directory:", os.getcwd())
print("Script location:", os.path.dirname(os.path.abspath(__file__)))
print("File exists:", os.path.exists("data/processed/positions.csv"))