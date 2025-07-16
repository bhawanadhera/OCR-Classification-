Dataset Link
OCR dataset obtained from [Kaggle] 
https://www.kaggle.com/datasets/harieh/ocr-dataset/data 


# OCR-Classification-
CODE 
import os 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix 
import kagglehub 

# Download latest version 
path = kagglehub.dataset_download("harieh/ocr-dataset") 
Expected Output 
print("Path to dataset files:", path) 
# Define the dataset path 
dataset_path = 
'/root/.cache/kagglehub/datasets/harieh/ocr-dataset/versions/1/dataset' 
# Initialize lists to hold data and labels 
data = [] 
labels = [] 
for label in os.listdir(dataset_path): 
label_path = os.path.join(dataset_path, label) 
if os.path.isdir(label_path): 
count = 0 
        for image_file in os.listdir(label_path): 
            image_path = os.path.join(label_path, image_file) 
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
            if image is None: 
                print(f"Skipped: {image_path}") 
                continue 
            image = cv2.resize(image, (32, 32)) 
            data.append(image) 
            labels.append(label) 
            count += 1 
 
 
# Convert to numpy arrays 
data = np.array(data) 
labels = np.array(labels) 
 
# ----------- Step 2: EDA ----------- 
print(f"Total images: {len(data)}") 
 
# Only proceed if data is not empty 
if len(data) > 0: 
    print(f"Image shape: {data[0].shape}") 
 
    plt.figure(figsize=(10, 5)) 
sns.countplot(x=labels, order=np.unique(labels)) 
plt.title('Class Distribution') 
plt.xlabel('Character') 
plt.ylabel('Number of Images') 
plt.xticks(rotation=90) 
plt.tight_layout() 
plt.show() 
else: 
print(" No images were loaded. Please verify the dataset path and structure.")

# ----------- Step 3: Prepare Data for ML ----------- 
# Flatten images and normalize 
data_flat = data.reshape((data.shape[0], -1)) / 255.0 
# Encode class labels 
le = LabelEncoder() 
labels_encoded = le.fit_transform(labels) 
# Train-test split 
X_train, X_test, y_train, y_test = train_test_split( 
data_flat, labels_encoded, test_size=0.2, random_state=42 
) 

# ----------- Step 4: Train and Evaluate Models ----------- 
# Helper to train and evaluate a model 
def train_evaluate(model, name): 
model.fit(X_train, y_train) 
preds = model.predict(X_test) 
print(f"\n{name} Classification Report:") 
print(classification_report(y_test, preds, target_names=le.classes_)) 
return preds 

# Train models 
svm_model = SVC() 
rf_model = RandomForestClassifier() 
lr_model = LogisticRegression(max_iter=1000) 
svm_preds = train_evaluate(svm_model, "SVM") 
rf_preds = train_evaluate(rf_model, "Random Forest") 
lr_preds = train_evaluate(lr_model, "Logistic Regression") 




# ----------- Step 5: Plot Confusion Matrices ----------- 
def plot_cm(y_true, y_pred, title): 
cm = confusion_matrix(y_true, y_pred) 
plt.figure(figsize=(10, 7)) 
sns.heatmap(cm, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_) 
plt.title(f'Confusion Matrix - {title}') 
plt.xlabel("Predicted") 
plt.ylabel("Actual") 
plt.tight_layout() 
plt.show() 
plot_cm(y_test, svm_preds, "SVM") 
plot_cm(y_test, rf_preds, "Random Forest") 
plot_cm(y_test, lr_preds, "Logistic Regression") 
  
 
 


 
