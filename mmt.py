import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy import stats  

from random import shuffle

from util import standardize
import models
from sklearn.model_selection import train_test_split

import xarray as xr 
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.stattools import durbin_watson



class mmt:
	'''
	class for performing post test MMT analysis


	'''

	def __init__(self, df, model="TBR", daily_incremental_spend=None, alpha=.05, model_type="Normal"):

		assert isinstance(df, pd.DataFrame), "Need to pass pandas dataframe"
		assert 'x' in df.columns , "Need x column in dataframe"
		assert 'y' in df.columns , "Need y column in dataframe"
		assert 'test_flag' in df.columns, "Need test flag in dataframe"
		assert df.test_flag.max() < 3 and df.test_flag.min() >= 0, "Test flag should be between 0-2"

		self.df = df 
		self.model_type = model_type

		self.model = model 
		# build in models (string) or a model instance that has a fit function and predict predict function.
		# take standardize outside of model class and change build_model to fit for model tranforance 


		self.ppf = stats.norm.ppf(alpha/2+(1-alpha)) # two-sided


		self.daily_incremental_spend = daily_incremental_spend
		if self.daily_incremental_spend is not None:
			self.daily_incremental_spend_cumm = self.daily_incremental_spend.cumsum()

		if 'x_spend' in df.columns and 'y_spend' in df.columns:
			self.version = 2 # build spend and KPI model
		else:
			self.version = 1 # just build KPI model


		self.model_instance = self.build_model(self.df[self.df.test_flag==0].x.values, self.df[self.df.test_flag==0].y.values )
		self.y_model = self.model_instance.y_s.inverse_transform(self.model_instance.predict(self.df[self.df.test_flag==0].x.values))

		if self.version == 2 and self.daily_incremental_spend is None: # build spend model as well
			self.spend_model_instance = self.build_model(self.df[self.df.test_flag==0].x_spend.values, self.df[self.df.test_flag==0].y_spend.values )
			self.spend_model = self.spend_model_instance.y_s.inverse_transform(self.spend_model_instance.predict(self.df[self.df.test_flag==0].x_spend.values))
		


	def build_model(self, x, y):
		'''
		pass the type of model to be built

		model instance just needs to return a predict function at a minimum. 

		returns class instance 
		'''
		if self.model == 'TBR' and self.model_type=="Normal":
			# build TBR model 
			return models.tbr(x, y)

		if self.model == 'TBR' and self.model_type=="Student":
			# build TBR model 
			return models.tbr(x, y, model_type=self.model_type)


	def impact(self):
		assert self.df.test_flag.max() >= 1, "Test flag needs a minimum of 1"

		self.y_counterfactual = self.model_instance.y_s.inverse_transform(self.model_instance.predict(self.df[self.df.test_flag==1].x.values))

		self.cumsum_arr = (np.array(self.df[self.df.test_flag==1].y).reshape(-1,1)-self.y_counterfactual.T ).T.cumsum(axis=1)


		self.mu = self.cumsum_arr.mean(axis=0)
		self.sd = self.cumsum_arr.std(axis=0)

		self.results = pd.DataFrame({
			"predicted cumulative" : self.y_counterfactual.mean(axis=0).sum() , 
			"actual cumulative" : self.df[self.df.test_flag==1].y.values.sum() , 
			"cumulative effect" : self.mu[-1],
			"lower 95"  : self.mu[-1] - self.ppf  * self.sd[-1],
			"upper 95"  : self.mu[-1]+ self.ppf * self.sd[-1], 
			"ci" : self.ppf  * self.sd[-1] } , index=["KPI"] )

		if self.version == 2 and self.daily_incremental_spend is None:
			self.spend_counterfactual = self.spend_model_instance.y_s.inverse_transform(self.spend_model_instance.predict(self.df[self.df.test_flag==1].x_spend.values))
			self.spend_cumsum_arr = (np.array(self.df[self.df.test_flag==1].y_spend).reshape(-1,1) - self.spend_counterfactual.T).T.cumsum(axis=1)

			# add spend incremental
			self.results = self.results.append(pd.DataFrame({
											  "predicted cumulative" :  self.spend_counterfactual.mean(axis=0).sum(), 
											  "actual cumulative" : self.df[self.df.test_flag==1].y_spend.sum(), 
											  "cumulative effect": self.spend_cumsum_arr[:,-1].mean(),
							 				  "lower 95"  : self.spend_cumsum_arr[:,-1].mean() - self.ppf  * self.spend_cumsum_arr[:,-1].std(),
											  "upper 95"  : self.spend_cumsum_arr[:,-1].mean() + self.ppf  * self.spend_cumsum_arr[:,-1].std(),
											  "ci" :   self.ppf  * self.spend_cumsum_arr[:,-1].std()} , index=["Spend"]))

			# add roms 

			self.roms_samples = (self.cumsum_arr-self.spend_cumsum_arr) / self.spend_cumsum_arr
			self.results =  self.results.append(pd.DataFrame({
											  "predicted cumulative" :  "NA", 
											  "actual cumulative" : "NA" , 
				 							  "cumulative effect": self.roms_samples[:,-1].mean(),
							 				  "lower 95"  : self.roms_samples[:,-1].mean()  - self.ppf  * self.roms_samples[:,-1].std(),
											  "upper 95"  : self.roms_samples[:,-1].mean()  + self.ppf  * self.roms_samples[:,-1].std(),
											  "ci" : self.ppf  * self.roms_samples[:,-1].std() } ,  index=["Roms"]))

		if self.version ==2 and self.daily_incremental_spend is not None:
			# add spend incremental
			self.results = self.results.append(pd.DataFrame({
											  "predicted cumulative" :  self.daily_incremental_spend.sum(), 
											  "actual cumulative" : self.daily_incremental_spend.sum(), 
											  "cumulative effect": self.daily_incremental_spend.sum(),
							 				  "lower 95"  : 'NA' ,
											  "upper 95"  : 'NA' ,
											  "ci": "NA"}, index=["Spend"]))
			# add roms
			self.results = self.results.append(pd.DataFrame({
											  "predicted cumulative" :  "NA", 
											  "actual cumulative" : "NA" , 
											  "cumulative effect": (self.mu[-1].mean()-self.daily_incremental_spend_cumm[-1]) / self.daily_incremental_spend_cumm[-1],
							 				  "lower 95"  : 'NA' ,
											  "upper 95"  : 'NA' ,
											  "ci" : "NA"}
											  , index=["Roms"]))

			self.roms_samples = (self.cumsum_arr-self.daily_incremental_spend_cumm) / self.daily_incremental_spend_cumm


	def plot_results(self, figsize=(10,5) ):
		if self.version == 1:
			fig, ax = plt.subplots(3,1,figsize=figsize)
		if self.version == 2:
			fig, ax = plt.subplots(4,1,figsize=figsize)

		mu = self.cumsum_arr.mean(axis=0)
		sd = self.cumsum_arr.std(axis=0)





		# model vs actual
		ax[0].plot(self.df.index , 
		        np.concatenate([  self.y_model.mean(axis=0) 
		                        , self.y_counterfactual.mean(axis=0) ] )
		        , color='red'
		        , label='model')


		ax[0].plot(self.df.index , 
		           self.df.y.values 
		        , color='blue'
		        , label='actual')


		ax[0].set_title("KPI Estimate and Actual")
		ax[0].axvline(self.df[self.df.test_flag==1].index[0], color='black')
		ax[0].axhline(0, color='black')
		ax[0].legend()


		# Point wise  
		ax[1].plot(self.df.index , 
		        np.concatenate([  (self.df[self.df.test_flag==0].y.values-self.y_model).mean(axis=0) 
		                        , (self.df[self.df.test_flag==1].y.values-self.y_counterfactual).mean(axis=0) ] )
		        , color='blue')


		ax[1].fill_between(self.df.index , 
		        np.concatenate([  (self.df[self.df.test_flag==0].y.values-self.y_model).mean(axis=0) 
		                        , (self.df[self.df.test_flag==1].y.values-self.y_counterfactual).mean(axis=0) ])
		        +self.ppf* np.concatenate([  (self.df[self.df.test_flag==0].y.values-self.y_model).std(axis=0) 
		                        , (self.df[self.df.test_flag==1].y.values-self.y_counterfactual).std(axis=0) ])
		        
		        , 
		        np.concatenate([  (self.df[self.df.test_flag==0].y.values-self.y_model).mean(axis=0) 
		                        , (self.df[self.df.test_flag==1].y.values-self.y_counterfactual).mean(axis=0) ])
		        -self.ppf* np.concatenate([  (self.df[self.df.test_flag==0].y.values-self.y_model).std(axis=0) 
		                        , (self.df[self.df.test_flag==1].y.values-self.y_counterfactual).std(axis=0) ])
		        
		        , color='coral')

		ax[1].set_title("Pointwise Difference KPI Estimate")
		ax[1].axvline(self.df[self.df.test_flag==1].index[0], color='black')
		ax[1].axhline(0, color='black')

		# cumulative 

		ax[2].plot(self.df.index , 
		        np.concatenate([(self.df[self.df.test_flag==0].y.values-self.y_model).mean(axis=0) , mu] )
		        , color='blue')


		ax[2].fill_between(self.df[self.df.test_flag==0].index
		                , (self.df[self.df.test_flag==0].y.values-self.y_model).mean(axis=0)-self.ppf *(self.df[self.df.test_flag==0].y.values-self.y_model).std(axis=0)
		                , (self.df[self.df.test_flag==0].y.values-self.y_model).mean(axis=0)+self.ppf *(self.df[self.df.test_flag==0].y.values-self.y_model).std(axis=0)
		                , color = 'coral')


		ax[2].fill_between(self.df[self.df.test_flag==1].index
		                , mu-self.ppf *sd
		                , mu+self.ppf *sd
		                , color = 'coral')

		ax[2].set_title("Cummulative Effect KPI Estimate")
		ax[2].axvline(self.df[self.df.test_flag==1].index[0], color='black')
		ax[2].axhline(0, color='black')

		if self.version == 2:
			ax[3].plot(self.df.index , 
			        np.concatenate([np.zeros(self.df[self.df.test_flag==0].shape[0]) , self.roms_samples.mean(axis=0)] )
			        , color='blue'
			        , label='RoMS Estimate')

			ax[3].fill_between(self.df[self.df.test_flag==1].index
			                , (self.roms_samples.mean(axis=0)-self.ppf *self.roms_samples.std(axis=0))  
			                , (self.roms_samples.mean(axis=0)+self.ppf *self.roms_samples.std(axis=0))
			                , color='coral'
			                , label='CI')

			ax[3].set_title("Cumulative RoMS Estimate")

			ax[3].axvline(self.df[self.df.test_flag==1].index[0], color='black')
			ax[3].axhline(0, color='black')
			ax[3].legend()




	def plot_cumm_impact_kpi(self , figsize=(10,5)):
		'''
		plot causal impact
		'''
		assert self.df.test_flag.max() >= 1, "Need test flag to be at least one and max of two."

		mu = self.cumsum_arr.mean(axis=0)
		sd = self.cumsum_arr.std(axis=0)

		fig, ax = plt.subplots(1,1,figsize=figsize)

		ax.plot(self.df[self.df.test_flag==1].index, mu)
		ax.fill_between(self.df[self.df.test_flag==1].index, mu-self.ppf *sd, mu+self.ppf *sd)
		ax.set_title("Cummulative Effect KPI Estimate")
		ax.axvline(self.df[self.df.test_flag==1].index[0])
		ax.axhline(0)

	def plot_cumm_impact_roms(self, figsize=(10,5)):
		'''
		plot cumulative causal impact on RoMS
		'''
		assert self.df.test_flag.max() >= 1, "Need test flag to be at least one and max of two."
		assert self.version == 2 or self.daily_incremental_spend, "Need to pass Test and Control spend with dataframe or pass daily incremental spend"

		if self.daily_incremental_spend is not None:
			fig, ax = plt.subplots(1,1,figsize=figsize)
			
			ax.plot(self.mu / self.daily_incremental_spend.cumsum())
			ax.fill_between(list(range(len(self.mu))), (self.mu-self.ppf *self.sd) / self.daily_incremental_spend.cumsum(), (self.mu+self.ppf *self.sd) / self.daily_incremental_spend.cumsum()), 
			ax.set_title("Cumulative RoMS Estimate")

		if self.daily_incremental_spend is None:
			fig, ax = plt.subplots(1,1,figsize=figsize)
			
			ax.plot(self.roms_samples.mean(axis=0))
			ax.fill_between(list(range(len(self.roms_samples.mean(axis=0)))), (self.roms_samples.mean(axis=0)-self.ppf *self.roms_samples.std(axis=0))  ,(self.roms_samples.mean(axis=0)+self.ppf *self.roms_samples.std(axis=0)) )
			ax.set_title("Cumulative RoMS Estimate")



	def aa_test(self, samples = 35):
		'''
		A/A Test: TBR: Do CV to estimate how often a sig change is detected in pre-period.. should be 5% or alpha parameter
		Breaks: Run cumsum test on residulals
		Residuals: check for normality and auto-correlation
		Model Fit: out of sampe RMSE, R2, etc.

		MDE: what is MDE KPI, Spend, ROMS at various spend and test lengths. 
		Fake Lift: same as MDE except simulate data versus using CI. 
		'''
		# seomthing
		self.rmse_bootstrap = []

		self.aa_kpi = []
		self.aa_spend = []

		if self.model == "TBR":
			df_aa = self.df[self.df.test_flag == 0].copy()
			for _ in range(samples):
				test_flag = np.concatenate([np.zeros(int(df_aa.shape[0]*.8)) , np.ones(int(df_aa.shape[0]*.2))])
				shuffle(test_flag)
				df_aa['test_flag'] = test_flag
				a1 = mmt(df_aa)
				a1.impact()
				self.aa_kpi.append(a1.results.loc['KPI']['absolute cummulative effect'])
				self.aa_spend.append(a1.results.loc['Spend']['absolute cummulative effect'])


	def mde(self, days = 40, samples=1):
		self.days = days
		if self.model=="TBR":

			self.mde_df = pd.DataFrame(columns = list( range(1, days+1)), index = np.linspace(.1, 1, 10))
			self.prc_df = pd.DataFrame(columns = ['percent', 'ci', 'cummulative'], index = list( range(1, days+1)))

			mda = mmt(self.df[self.df.test_flag==0])

			y_counterfactual = mda.model_instance.y_s.inverse_transform(mda.model_instance.predict(self.df[self.df.test_flag==0].x.values))
			if self.version == 2:
				spend_counterfactual = mda.spend_model_instance.y_s.inverse_transform(mda.spend_model_instance.predict(self.df[self.df.test_flag==0].x_spend.values))

			for _ in range(samples):
				for d in range(1, days+1):
					for i in np.linspace(.1, 1, 10):
						random_indices = np.random.choice(y_counterfactual.shape[1],  size=d,  replace=True)

						ci_50 = y_counterfactual[:,random_indices].sum(axis=1).std()*self.ppf

						if self.version == 2:
							increase = np.mean(spend_counterfactual[:,random_indices].sum(axis=1)*i)

							if not isinstance(self.mde_df.loc[i][d], list):
							    self.mde_df.loc[i][d] = []
							    
							if isinstance(self.mde_df.loc[i][d], list):
							    self.mde_df.loc[i][d].append(ci_50 / increase) 


			for d in range(1, days+1):
				for i in np.linspace(.1, 1, 10):
					random_indices = np.random.choice(y_counterfactual.shape[1],  size=d,  replace=True)
					ci_50 = y_counterfactual[:,random_indices].sum(axis=1).std()*self.ppf
					self.prc_df.loc[d]['percent'] = ci_50 / y_counterfactual[:,random_indices].sum(axis=1).mean()
					self.prc_df.loc[d]['ci'] = ci_50  
					self.prc_df.loc[d]['cummulative'] =  y_counterfactual[:,random_indices].sum(axis=1).mean()

			for d in range(1, days+1):
				for i in np.linspace(.1, 1, 10):
					m, s = np.mean(self.mde_df.loc[i][d]), np.std(self.mde_df.loc[i][d])
					self.mde_df.loc[i][d] = [m,s]

		if self.version == 2:
			return self.mde_df, self.prc_df
		if self.version == 1:
			return self.prc_df

	def fake_lift(self, points=100, max_effect = .1, days=40):
		# kpi
		prob = []

		effect = np.linspace(-max_effect, max_effect, points)

		if self.model=="TBR":
			df_fake_lift = self.df[self.df.test_flag==0][:-days]


			mda = mmt(self.df[self.df.test_flag==0])
			y_counterfactual = mda.model_instance.y_s.inverse_transform(mda.model_instance.predict(self.df[self.df.test_flag==0].x.values))
			# spend_counterfactual = mda.spend_model_instance.y_s.inverse_transform(mda.spend_model_instance.predict(self.df[self.df.test_flag==0].x_spend.values))

			for lift in effect:
				y_fake = self.df[self.df.test_flag==0][-days:].y.values * lift + self.df[self.df.test_flag==0][-days:].y.values 
				y_fake_cum = y_fake.sum()

				y_counterfactual = mda.model_instance.y_s.inverse_transform(mda.model_instance.predict(self.df[self.df.test_flag==0][-days:].x.values))


				incremental = y_fake - y_counterfactual

				if lift >= 0:
					p_effect = ((incremental.sum(axis=1)>0)+0).sum() / incremental.sum(axis=1).shape[0]
				if lift < 0:
					p_effect = ((incremental.sum(axis=1)<0)+0).sum() / incremental.sum(axis=1).shape[0]

				prob.append(p_effect)


		fake_lift = list(zip(effect, prob))
		psuedo_powr_curve = pd.DataFrame(fake_lift)
		psuedo_powr_curve.columns = ['fake_lift', 'Prob. of Causal Inference']

		# plot power curve 
		plt.plot(psuedo_powr_curve['fake_lift'], psuedo_powr_curve['Prob. of Causal Inference'],  marker='o')
		plt.title("Power Curve")
		plt.xlabel("Fake Lift")
		plt.ylabel("Probability of Causal Inference: KPI")

		return psuedo_powr_curve


	def residual_analysis(self):
		self.residuals = (self.df[self.df.test_flag==0].y.values-self.y_model).mean(axis=0)
		k2, k2_p_value = stats.normaltest(self.residuals)
		s, s_p_value = stats.shapiro(self.residuals)
		dw = durbin_watson(self.residuals)
		b, p, d = breaks_cusumolsresid(self.residuals)

		residual_analysis = pd.DataFrame({'normality_test' : [k2_p_value, 'accept' if k2_p_value > .05 else 'reject']
		                                  , 'shaprio' : [s_p_value, 'accept' if s_p_value > .05 else 'reject']
		                                  , 'durbin_watson' : [dw, 'accept' if dw >= 1.5 and dw <= 2.5 else 'reject']
		                                  , 'breaks' : [p, 'accept' if p > .05 else 'reject'] }
		                                  ,  index = ['p_values_kpi', 'accept_reject_kpi'] 
		                                  )

		if self.version == 2:
			residuals_spend = (self.df[self.df.test_flag==0].y_spend.values-self.spend_model).mean(axis=0)
			k2, k2_p_value = stats.normaltest(residuals_spend)
			s, s_p_value = stats.shapiro(residuals_spend)
			dw = durbin_watson(residuals_spend)
			b, p, d = breaks_cusumolsresid(residuals_spend)

			spend_ra = pd.DataFrame({'normality_test' : [k2_p_value, 'accept' if k2_p_value > .05 else 'reject']
                                  , 'shaprio' : [s_p_value, 'accept' if s_p_value > .05 else 'reject']
                                  , 'durbin_watson' : [dw, 'accept' if dw >= 1.5 and dw <= 2.5 else 'reject']
                                  , 'breaks' : [p, 'accept' if p > .05 else 'reject'] }
                                  ,  index = ['p_values_spend', 'accept_reject_spend'] 
                                  )

			residual_analysis = pd.concat([residual_analysis, spend_ra])

		return residual_analysis