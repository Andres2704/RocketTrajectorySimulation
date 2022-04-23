import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

class GaussianFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N) # Mu
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0]) # Stdev
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1) # Create gaussian basis functions

def wind_component(M, h, X_fit, data):
    ml_model = make_pipeline(GaussianFeatures(M), LinearRegression())
    ml_model.fit(h[:, np.newaxis], data)
    yfit = ml_model.predict(X_fit[:, np.newaxis])
    return yfit
    
# Function for calculating the altitude based on pressure value and ground properties
def calc_high(P):
    # P in mBar
    L = -6.5E-3
    T0 = 288.15
    P0 = 1
    R = 287
    g = 9.81
    P = P/1013
    return (T0/L)*(((P/P0)**(-L*R/g)) - 1)
