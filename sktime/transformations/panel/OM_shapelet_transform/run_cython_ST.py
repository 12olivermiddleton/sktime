
from sktime.utils.validation.panel.OM_shapelet_transform import cython_shapelet_transform
from sktime.datasets import load_unit_test

# X_train, y_train = load_unit_test(split="train", return_X_y=True)
# t = RandomShapeletTransform(
#     n_shapelet_samples = 500,
#     max_shapelets = 10,
#     batch_size = 100,)
#
# t.fit(X_train, y_train)
# X_t = t.transform(X_train)