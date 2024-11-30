import joblib
from sklearn.linear_model import LinearRegression
import unittest
import numpy as np

class TestModel(unittest.TestCase):
    def test_model(self):
        model = joblib.load('model/house.pkl')
        self.assertIsInstance(model, LinearRegression)
        # Check if model.coef_ is a 1D or 2D array
        if isinstance(model.coef_, np.ndarray):
            if model.coef_.ndim == 1:
                self.assertGreaterEqual(len(model.coef_), 1)
            elif model.coef_.ndim == 2:
                self.assertGreaterEqual(len(model.coef_[0]), 1)
        else:
            self.fail("model.coef_ is not a numpy array")

if __name__ == '__main__':
    unittest.main()