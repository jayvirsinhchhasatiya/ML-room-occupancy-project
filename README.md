
# Room Occupancy Detection using Machine Learning

## Problem Statement

To identify whether there is any person present in a room or not




## Authors

- [@jayvirsinhchhasatiya](https://www.github.com/jayvirsinhchhasatiya)
- [@Dhwani-48](https://www.github.com/Dhwani-48)


## About Problem

- Identify occupancy
- Bussiness Understanding
  - Application
    - Military operation
    -	To see if there is thief in the house 
    - Save energy
- Dataset
  - Available in model folder


## Setting python environment

- We use python for creating ML model

### Importing the Dependencies

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
```

### Loading Dataset

```python
df = pd.read_csv("C:\\Users\\JAYVIR CHHASATIYA\\Desktop\\Data science\\case-study-june-2022 (1)\\data\\occupancy.csv")

```


