#!/usr/bin/python

import pandas as pd

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from tools.feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from utils import get_original_data, get_k_best_features


# Task 0: Preview and prepare the data.
# -----------------------------------------------------------------------------------------
data = get_original_data()
df = pd.DataFrame.from_dict(data).transpose()

# Some field need to be converted as numeric type
df = df.convert_objects(convert_numeric=True)

# Task 1: Select what features you'll use.
# -----------------------------------------------------------------------------------------

initial_feature_list = ['bonus', 'exercised_stock_options', 'expenses',
                        'from_messages', 'from_poi_to_this_person',
                        'from_this_person_to_poi', 'long_term_incentive', 'other',
                        'restricted_stock', 'salary', 'shared_receipt_with_poi',
                        'to_messages', 'total_payments', 'total_stock_value']


# Task 2: Remove outliers
# -----------------------------------------------------------------------------------------

# Remove TOTAL, This is total for other field.
df.drop(df.index[df.index == 'TOTAL'], inplace=True)

# Remove THE TRAVEL AGENCY IN THE PARK, Does not belong in the data.
df.drop(df.index[df.index == 'THE TRAVEL AGENCY IN THE PARK'], inplace=True)

# Remove LOCKHART EUGENE E, Only a single field has data, the rest is NaN.
df.drop(df.index[df.index == 'LOCKHART EUGENE E'], inplace=True)


# Task 3: Create new feature(s)
# -----------------------------------------------------------------------------------------

# Fraction from POI
df['fraction_from_poi'] = df.apply(lambda x: x['from_poi_to_this_person'] / x['to_messages'], axis=1)
# Fraction to POI
df['fraction_to_poi'] = df.apply(lambda x: x['from_this_person_to_poi'] / x['from_messages'], axis=1)

initial_feature_list += ['fraction_from_poi', 'fraction_to_poi']

# Transform the data back to dictionary format from Panda Dataframe
my_dataset = df.fillna('NaN').transpose().to_dict()

# Task 4: Run a classifier
# -----------------------------------------------------------------------------------------

# Let's reduce the features to 4 for now.
kbest_features = get_k_best_features(my_dataset, ['poi'] + initial_feature_list)
kbest_features = [item[0] for item in kbest_features[:4]]

# The first feature must be "poi".
features_list = ['poi'] + kbest_features

# Setup label and features
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# Task 5: Tune classifier
# -----------------------------------------------------------------------------------------

# We will be using GridSearchCV to find the optimum
# parameters.
pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', KNeighborsClassifier())
])

# Parameters to be tuned
metrics = ['minkowski', 'euclidean', 'manhattan']
weights = ['uniform', 'distance']
n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
param_grid_knc = dict(classifier__metric=metrics,
                      classifier__weights=weights,
                      classifier__n_neighbors=n_neighbors)

# Setup cross validation
cv = StratifiedShuffleSplit(labels, 1000, random_state=42)

# Run the GridSearch
clf_knc = GridSearchCV(pipe, param_grid=param_grid_knc, cv=cv, scoring='recall')
clf_knc.fit(features, labels)

clf = clf_knc.best_estimator_


# Task 6: Check result
# -----------------------------------------------------------------------------------------

dump_classifier_and_data(clf, my_dataset, features_list)
