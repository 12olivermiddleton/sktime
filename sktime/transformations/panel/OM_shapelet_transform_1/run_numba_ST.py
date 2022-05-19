# This python file is for running the Numba transform algorithm
# Oliver Middleton

# The following script is taken from the shapelet transform comments,
# and can be run once instead of using the python console each time
import time


from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform
from sktime.datasets import load_gunpoint  # This can be any of the baked in datasets

time1 = time.time()
X_train, y_train = load_gunpoint(split="train", return_X_y=True)
t = RandomShapeletTransform(n_shapelet_samples=500, max_shapelets=10, batch_size=100)
t.fit(X_train, y_train)
X_t = t.transform(X_train)
time2 = time.time()

print("Total elapsed time: ", time2-time1)