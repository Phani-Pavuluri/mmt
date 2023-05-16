"""
This module contains helper functions and classes

update to make follow sklearn method convection: fit, fit_transform, inverse transform 

"""

import numpy as np 


class standardize:
	"""
	Class for standardizing a numpy array
	x = numpy array

	TODO:
		add qa
		add assert to check conditions are meant 

	Notes: 
		CB: I prefer this to sklearn's standardize. I can't remember why as I wrote this piece of code a long time ago. Not opposed to using sklearn if that makes more sense.
	"""

	def __init__(self, x):
		self.x = x 
		self.std = np.std(x)
		self.mean = np.mean(x)
	
	def transform(self, x):
		return ( x - self.mean ) / self.std
	
	def inverse_transform(self, z):
		return z * self.std + self.mean


def acorr_bg(x, y):
    '''
    Get's p-value for acorr_breusch_godfrey test
    TODO: find formalua and contruct class to get with PM Linear model instead of running sm.OLS
    '''
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()

    bg = acorr_breusch_godfrey(model, nlags=3)
    return bg

# build out of sample rmse function