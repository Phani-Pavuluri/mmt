""" Models for MMT """


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

from util import standardize

import xarray as xr
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.stattools import durbin_watson

class tbr:
	'''
	class to build and store simple bayesian linear model

	build linear model with pre-test data.
	'''

	def __init__(self, x, y, model_type="Normal", **params):
		self.x_s = standardize(x)  # creates class with standarization info
		self.y_s = standardize(y) # hello

		self.x = self.x_s.transform(x)  # standardizes array
		self.y = self.y_s.transform(y)
		self.model_type = model_type

		self.model, self.idata = self.simple_linear(self.x, self.y, coords=pd.Series(range(len(self.x)), name="t"))  # builds simple linear model

		self.idata.extend(pm.sample_prior_predictive(model=self.model))
		self.idata.extend(pm.sample_posterior_predictive(self.idata, model=self.model))

		self.idata.posterior["y_model"] = self.idata.posterior["intercept"] + self.idata.posterior["x_coeff"] * xr.DataArray(self.x )

		

	def simple_linear(self, x, y, coords, **params):

		if self.model_type == "Normal":
			with pm.Model() as model:

				# set data
				x = pm.MutableData("x", x , dims="t", coords = coords)
				y = pm.MutableData("y", y , dims="t", coords = coords)
				
				# Define priors
				sigma = pm.HalfNormal("sigma", 1  )
				intercept = pm.Normal("intercept", 0, sigma=1 )
				x_coeff = pm.Normal("x_coeff", 0, sigma=1 )

				# Define likelihood
				likelihood = pm.Normal("y_hat", mu=intercept + x_coeff * self.x, sigma=sigma, observed=self.y, dims="t")

				# Inference!
				# draw 3000 posterior samples using NUTS sampling

				idata = pm.sample(3000)

			return model, idata

		if self.model_type == "Student":
			with pm.Model() as model:

				# set data
				x = pm.MutableData("x", x , dims="t", coords = coords)
				y = pm.MutableData("y", y , dims="t", coords = coords)
				
				# Define priors
				sigma = pm.HalfNormal("sigma", 1  )
				intercept = pm.Normal("intercept", 0, sigma=1 )
				x_coeff = pm.Normal("x_coeff", 0, sigma=1 )

				nu = pm.Uniform('nu', lower=1, upper=100)

				# Define likelihood
				likelihood = pm.StudentT("y_hat", mu=intercept + x_coeff * self.x, sigma=sigma,  nu=nu, observed=self.y, dims="t")

				# Inference!
				# draw 3000 posterior samples using NUTS sampling

				idata = pm.sample(3000)

			return model, idata



	def plot_relationships(self):
		plt.figure(figsize=(10,5))
		plt.plot(self.x)
		plt.plot(self.y)

		fig, axs = plt.subplots(1,2 , figsize=(10,5))


		import warnings 

		def fxn():
		    warnings.warn("deprecated", DeprecationWarning)

		with warnings.catch_warnings():
		    warnings.simplefilter("ignore")
		    fxn()


		    az.plot_lm(idata=self.idata, y="y_hat", x=self.x, axes=axs[0])

		    az.plot_lm(idata=self.idata \
		               , y="y_hat"\
		               , x="x" 
		               , y_model="y_model"\
		               , kind_pp="hdi"\
		               , y_hat_fill_kwargs={'hdi_prob':.9}
		              , axes=axs[1])

		    axs[0].set_title("Posterior Predictive")
		    axs[1].set_title("Posterior Predictive: 90% HDI")


	def predict(self, x, samples=1000):

		x = self.x_s.transform(x)

		y_hat = []

		chains = self.idata.posterior.dims['chain']
		draws  = self.idata.posterior.dims['draw']


		for _ in range(samples):
			intercept_s = self.idata.posterior['intercept'].sel(chain=np.random.randint(chains), draw=np.random.randint(draws))
			beta_1_s = self.idata.posterior['x_coeff'].sel(chain=np.random.randint(chains), draw=np.random.randint(draws))
			sigma_s = self.idata.posterior['sigma'].sel(chain=np.random.randint(chains), draw=np.random.randint(draws))

			y_hat.append(np.random.normal( intercept_s.values + beta_1_s.values * x, sigma_s.values))

		return np.array(y_hat) 
