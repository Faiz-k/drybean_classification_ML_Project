from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Create your views here.
def home(request):
    if request.method == 'POST':
        # Retrieve values for each field
        area = request.POST.get('area')
        perimeter = request.POST.get('perimeter')
        majorAxisLength = request.POST.get('majorAxisLength')
        minorAxisLength = request.POST.get('minorAxisLength')
        aspectRatio = request.POST.get('aspectRatio')
        eccentricity = request.POST.get('eccentricity')
        convexArea = request.POST.get('convexArea')
        equivDiameter = request.POST.get('equivDiameter')
        extent = request.POST.get('extent')
        solidity = request.POST.get('solidity')
        roundness = request.POST.get('roundness')
        compactness = request.POST.get('compactness')
        shapeFactor1 = request.POST.get('shapeFactor1')
        shapeFactor2 = request.POST.get('shapeFactor2')
        shapeFactor3 = request.POST.get('shapeFactor3')
        shapeFactor4 = request.POST.get('shapeFactor4')
        data = pd.read_csv('C:\\Users\\mf879\\OneDrive\\Desktop\\6_drybeanclassification\\train_dataset.csv')

# Split the dataset into features (X) and target labels (y)
        X = data.drop('Class','columns')
        y = data['Class']

# Split the data into a training set and a testing set (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values (mean=0, std=1)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

# Choose the number of neighbors (k)
        k = 5

# Create and train the KNN classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
        y_pred = knn_classifier.predict([[area,	perimeter	,majorAxisLength,minorAxisLength,aspectRatio,eccentricity,convexArea, equivDiameter,	extent,	solidity	,roundness,	compactness,	shapeFactor1,	shapeFactor2,	shapeFactor3,	shapeFactor4]])
        

        
        return render(request, "home.html", context={'y_pred':y_pred })

    return render(request, 'home.html')
