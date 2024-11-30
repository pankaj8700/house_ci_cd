import joblib
from sklearn.linear_model import LinearRegression
import unittest

class TestModel(unittest.TestCase):
    def test_model(self):
        model = joblib.load('model/house.pkl')
        self.assertIsInstance(model, LinearRegression)
        self.assertGreaterEqual(len(model.coef_[0]), 8)

if __name__ == '__main__':
    unittest.main()