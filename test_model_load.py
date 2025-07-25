import pickle

with open(r'C:\Users\nikch\OneDrive\Desktop\Heatlh Readmission\Heatlh Readmission\model\readmission_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully!")

