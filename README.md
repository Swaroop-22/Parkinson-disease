
# Parkinson's Disease Detection using Support Vector Machines (SVM)

This repository features an end-to-end Machine Learning pipeline designed to detect the presence of Parkinson's Disease in patients using biometric voice frequency measurements. By analyzing acoustic properties extracted from speech recordings, the model acts as a non-invasive diagnostic decision tool to classify individuals as either Healthy or Parkinson's Positive.

---

## 📌 Project Architecture & Workflow

The workflow uses statistical data profiling and supervised classification to achieve accurate predictive analysis:

1. **Exploratory Data Analysis (EDA):** Profiles a clinical dataset containing 195 records across 24 specific biomedical voice traits.
2. **Data Separation:** Isolates subject metadata identifiers (`name`) and targets (`status`) away from the mathematical training arrays ($X$).
3. **Data Standardization:** Applies feature scaling via `StandardScaler` to transform disparate acoustic ranges into a uniform Gaussian distribution ($\mu = 0, \sigma = 1$). This step is critical to prevent features with larger scales from dominating the calculation of the Support Vector boundary.
4. **Supervised Classification:** Trains a linear Support Vector Machine (`svm.SVC`) to locate the optimal hyper-plane maximizing the structural margin between healthy and symptomatic vectors.

---

## 📊 Dataset Feature Profile

The dataset (`parkinsons.csv`) maps individual vocal metrics to binary health states:
- **`status = 1`:** Parkinson's Positive (147 instances)
- **`status = 0`:** Healthy (48 instances)

### Core Acoustic Measurements Include:
* **`MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`:** Average, maximum, and minimum fundamental vocal frequencies.
* **`MDVP:Jitter(%)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`:** Multiple variations of frequency parameter variations (measures of voice tone pitch instability).
* **`MDVP:Shimmer`, `Shimmer:APQ3`, `Shimmer:DDA`:** Structural measures of sound amplitude variations (measures of voice volume instability).
* **`NHR`, `HNR`:** Noise-to-Harmonics and Harmonics-to-Noise ratios, characterizing acoustic clarity vs. speech breathiness.
* **`spread1`, `spread2`, `PPE`:** Non-linear fundamental frequency variation metrics.

---

## 🛠️ Installation & Dependencies

To execute the code scripts or view development changes locally, configure your workspace with the following scientific data libraries:

```bash
pip install numpy pandas scikit-learn

```

---

## 💻 Step-by-Step Pipeline Implementation

### 1. Preprocessing & Data Scaling

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('parkinsons.csv')

# Drop identifier attributes and separate target arrays
X = data.drop(columns=['name', 'status'], axis=1)
y = data['status']

# Split into train/test sets with random seed protection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Fit scaler only to the training split to avoid data leakage
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

```

### 2. SVM Model Training & Evaluation

```python
from sklearn import svm
from sklearn.metrics import accuracy_score

# Initialize a Linear Support Vector Classifier
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate performance metrics
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print(f"Accuracy Score of Training Data: {accuracy_score(y_train, train_preds):.4f}")
print(f"Accuracy Score of Testing Data: {accuracy_score(y_test, test_preds):.4f}")

```

---

## 📈 Metric Diagnostic Output

The structural model exhibits balanced generalization properties across unseen data partitions, proving that it avoids extreme variance issues:

* **Training Classification Accuracy:** **88.46%**
* **Testing/Validation Accuracy:** **87.18%**

---

## 🔮 Predictive Diagnostic System

The application features a real-time single-instance validation pipeline. Raw, unstructured telemetry arrays are reshaped, scaled, and inferred instantly using the trained pipeline components:

```python
import numpy as np

# Sample raw feature values input array
input_data = (197.076, 206.896, 192.055, 0.00289, 0.00001, 0.00166, 0.00168, 
              0.00498, 0.01098, 0.09700, 0.00563, 0.00680, 0.00802, 0.01689, 
              0.00339, 26.77500, 0.422229, 0.741367, -7.348300, 0.177551, 1.743867, 0.085569)

# Convert to numpy array and reshape for single-instance prediction
in_array = np.asarray(input_data).reshape(1, -1)

# Apply saved standardization parameters
std_instance = scaler.transform(in_array)

# Model check
prediction = model.predict(std_instance)

if prediction[0] == 0:
    print('Diagnosis: Individual is Healthy.')
else:
    print('Diagnosis: Individual is Parkinson Positive (+ve).')

```

---

## 📝 Future Scope

* **Hyperparameter Fine-Tuning:** Integrate `GridSearchCV` to experiment with Alternative Kernels like Radial Basis Functions (`kernel='rbf'`) and tune penalty variables ($C$).
* **Handling Class Imbalance:** Since the positive sample rate is high (~75%), introducing synthetic data balancing strategies (e.g., SMOTE) could enhance precision boundaries for healthy patients.

```

```
