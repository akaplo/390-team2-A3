# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:02:58 2016

@author: cs390mb

Assignment 3 : Speaker Identification

This is the starter script for training a model for identifying
speaker from audio data. The script loads all labelled speaker
audio data files in the specified directory. It extracts features
from the raw data and trains and evaluates a classifier to identify
the speaker.

"""
# Makes float division default
from __future__ import division

import os
import sys
import numpy as np

# The following are classifiers you may be interested in using:
from sklearn.tree import DecisionTreeClassifier # decision tree classifier
from sklearn.ensemble import RandomForestClassifier # random forest classifier
from sklearn.neighbors import NearestNeighbors # k-nearest neighbors (k-NN) classiifer
from sklearn.svm import SVC #SVM classifier

from features import FeatureExtractor
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import pickle

# I prefer train test split to kfold
from sklearn.cross_validation import train_test_split

# Needed for preprocessing
from sklearn import preprocessing

# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

data_dir = 'data' # directory where the data files are stored

output_dir = 'training_output' # directory where the classifier(s) are stored

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# the filenames should be in the form 'speaker-data-subject-1.csv', e.g. 'speaker-data-Erik-1.csv'. If they
# are not, that's OK but the progress output will look nonsensical

class_names = [] # the set of classes, i.e. speakers

data = np.zeros((0,8002)) #8002 = 1 (timestamp) + 8000 (for 8kHz audio data) + 1 (label)

for filename in os.listdir(data_dir):
    if filename.endswith(".csv") and filename.startswith("speaker-data"):
        filename_components = filename.split("-") # split by the '-' character
        speaker = filename_components[2]
        print("Loading data for {}.".format(speaker))
        if speaker not in class_names:
            class_names.append(speaker)
        speaker_label = class_names.index(speaker)
        sys.stdout.flush()
        data_file = os.path.join('data', filename)
        data_for_current_speaker = np.genfromtxt(data_file, delimiter=',')
        print("Loaded {} raw labelled audio data samples.".format(len(data_for_current_speaker)))
        sys.stdout.flush()
        data = np.append(data, data_for_current_speaker, axis=0)

print("Found data for {} speakers : {}".format(len(class_names), ", ".join(class_names)))

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

# You may need to change n_features depending on how you compute your features
# we have it set to 3 to match the dummy values we return in the starter code.
n_features = 997

print("Extracting features and labels for {} audio windows...".format(data.shape[0]))
sys.stdout.flush()

X = np.zeros((0,n_features))
y = np.zeros(0,)

# change debug to True to show print statements we've included:
feature_extractor = FeatureExtractor(debug=False)

for i,window_with_timestamp_and_label in enumerate(data):
    window = window_with_timestamp_and_label[1:-1] # get window without timestamp/label
    label = data[i,-1] # get label
    x = feature_extractor.extract_features(window)  # extract features

    # if # of features don't match, we'll tell you!
    if (len(x) != X.shape[1]):
        print("Received feature vector of length {}. Expected feature vector of length {}.".format(len(x), X.shape[1]))

    X = np.append(X, np.reshape(x, (1,-1)), axis=0)
    y = np.append(y, label)

print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()


# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

n = len(y)
n_classes = len(class_names)

# This is all of the actual labels for the classes, needed for confusion matrix
class_labels = list(set(y))

# Split the data into a training set and a test set (evaluation set, hence val)
X_t, X_val, y_t, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# Create a scaler
scaler = preprocessing.StandardScaler().fit(X_t)

# Scale both data sets
X_t = scaler.transform(X_t)
X_val = scaler.transform(X_val)

# This is how the best classifier was selected
"""
fits = []

for max_feature in [100, 500, 900]:
    for max_depth in [5, 10, 15]:
        for n_estimator in [10, 30, 50]:
            clf = RandomForestClassifier(n_estimators=n_estimator, max_features=max_feature, max_depth=max_depth)
            clf.fit(X_t, y_t)
            conf = confusion_matrix(clf.predict(X_val), y_val, labels=class_labels)
            fits += [({'max_features': max_feature, 'max_depth': max_depth, 'n_estimators': n_estimator}, sum(sum(np.multiply(conf, np.eye(n_classes)))) / sum(sum(conf)))]

print fits
"""

# Initialize the classifier
clf = RandomForestClassifier(n_estimators=50, max_features=500, max_depth=15)

# Train the classifier on the preprocessed training data
clf.fit(X_t, y_t)

# Make a confusion matrix
conf = confusion_matrix(clf.predict(X_val), y_val, labels=class_labels)

# Print Accuracy, precision, and recall
print "average accuracy: %s"%(sum(sum(np.multiply(conf, np.eye(n_classes)))) / sum(sum(conf)))
print "average precision: %s"%([conf[i, i] / sum(conf[:, i]) for i in range(0, n_classes)])
print "average recall: %s"%([conf[i, i] / sum(conf[i, :]) for i in range(0, n_classes)])

# The best classifier was already selected but I stuck with your paradigm
best_classifier = clf

# Train the best classifier with all of the data, scaled
best_classifier.fit(scaler.transform(X),y)

classifier_filename='classifier.pickle'
print("Saving best classifier to {}...".format(os.path.join(output_dir, classifier_filename)))
with open(os.path.join(output_dir, classifier_filename), 'wb') as f: # 'wb' stands for 'write bytes'
    pickle.dump(best_classifier, f)

# This serializes the scaler so it can be used in spealer-identification.py
scaler_filename='scaler.pickle'
print("Saving scaler to {}...".format(os.path.join(output_dir, classifier_filename)))
with open(os.path.join(output_dir, scaler_filename), 'wb') as f: # 'wb' stands for 'write bytes'
    pickle.dump(scaler, f)
