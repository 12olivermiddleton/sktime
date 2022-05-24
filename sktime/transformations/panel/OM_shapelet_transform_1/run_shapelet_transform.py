# This python file is for running the shapelet transform algorithm
# Oliver Middleton

import time


from sktime.transformations.panel.OM_shapelet_transform_1.base_shapelet_transform import RandomShapeletTransform
from sktime.datasets import load_UCR_UEA_dataset  # This can be any of the baked in datasets
from sktime.datasets import load_acsf1,load_basic_motions,  load_unit_test, load_gunpoint, load_osuleaf

# Data sets for load_UCR_UEA_dataset
data = [
    "ACSF1",
    "Adiac",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "Car",
    "CBF",
    "Coffee",
    "DiatomSizeReduction",
    "ECG200",
    "EthanolLevel",
    "FiftyWords",
    "Fish",
    "FreezerRegularTrain",
    "Fungi",
    "Herring",
    "Mallat",
    "Meat",
    "MedicalImages",
    "OliveOil",
    "PigAirwayPressure",
    "Plane",
    "Rock",
    "SyntheticControl",
    "Wine",
    "Yoga",
]

data_times = []
# loop through data sets and run numba shapelet transform implementation on each
for set in data:
    time1 = time.time()
    print(set)
    X_train, y_train = load_UCR_UEA_dataset(set, return_X_y=True)
    t = RandomShapeletTransform(
        n_shapelet_samples=500,
        max_shapelets=10,
        batch_size=100, )

    t.fit(X_train, y_train)
    X_t = t.transform(X_train)

    time2 = time.time()
    total = time2 - time1

    print("Total elapsed time: ", float("{:.2f}".format(total)))
    data_times.append(float("{:.2f}".format(total)))

print(data_times)