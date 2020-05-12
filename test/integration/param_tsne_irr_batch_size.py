from dimdrop.models import ParametricTSNE
import numpy as np

# Parametric t-SNE with a batch size that is not a multiple of the length of
# the data
arr = np.random.rand(100, 3)
model = ParametricTSNE(3, 2, batch_size=33, verbose=True)
test = model.fit_transform(arr)
