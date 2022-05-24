import time

from sktime.datasets import load_acsf1, load_basic_motions, load_gunpoint, load_osuleaf, load_unit_test
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.pipeline import Pipeline


# Uncomment the implementation you wish to test the accuracy for
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform
# from sktime.transformations.panel.OM_shapelet_transform_1.base_shapelet_transform import RandomShapeletTransform
# from sktime.transformations.panel.OM_shapelet_transform_1.base_shapelet_transform import RandomShapeletTransform


# Change the  function load_acsf1 to any of the loaded in data sets
train_x, train_y = load_acsf1(split='train', return_X_y=True)
test_x, test_y = load_acsf1(split='test', return_X_y=True)

time_contract_in_mins = 1

#  Contracted Pipeline with 1 minute time limit
pipeline = Pipeline([
    ('st', RandomShapeletTransform(n_shapelet_samples=500,
                                   max_shapelets=10,
                                   batch_size=100, )),
    ('rf', RandomForestClassifier(n_estimators=100)),
])

start = time.time()
pipeline.fit(train_x, train_y)
end_build = time.time()
preds = pipeline.predict(test_x)
end_test = time.time()

print("Results:")
print("Correct:")
correct = sum(preds == test_y)
print("\t" + str(correct) + "/" + str(len(test_y)))
print("\t" + str(correct / len(test_y)))
print("\nTiming:")
print("\tTo build:   " + str(end_build - start) + " seconds")
print("\tTo predict: " + str(end_test - end_build) + " seconds")
