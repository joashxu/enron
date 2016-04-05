#!/usr/bin/python
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest

from tools.feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, load_classifier_and_data, test_classifier


def setup_and_test(my_dataset, features_list, classifier, evaluation_func=None):
    # Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    # Setup classifier
    clf = classifier

    # Evaluate
    if not evaluation_func:
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=42)

    # Dump classifier and features list, so we can test them
    dump_classifier_and_data(clf, my_dataset, features_list)

    # load up student's classifier, dataset, and feature_list
    clf, dataset, feature_list = load_classifier_and_data()
    # Run testing script
    test_classifier(clf, dataset, feature_list)

    return


def get_original_data():
    """
    Unpickle data and return it.
    """
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    return data_dict


def computeFraction(poi_messages, all_messages):
    """
    Given a number messages to/from POI (numerator)
    and number of all messages to/from a person (denominator),
    return the fraction of messages to/from that person
    that are from/to a POI
    """
    if not poi_messages == 'NaN' and not all_messages == 'NaN':
        return poi_messages / float(all_messages)
    return 0.


def get_k_best_features(data, features_list, k):
    # Setup the label and features
    data = featureFormat(data, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    # Apply SelectKBest
    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_

    # pair up with feature name, ignore the first one, since
    # that is the 'poi' label
    unsorted_pairs = zip(features_list[1:], scores)

    # Sort based on score
    sorted_pairs = list(sorted(unsorted_pairs, key=lambda x: x[1], reverse=True))

    return sorted_pairs