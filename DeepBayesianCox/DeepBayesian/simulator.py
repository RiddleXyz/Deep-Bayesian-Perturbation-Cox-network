"""
This file is inherited from DeepSurv for generating simulated survival data.
"""
from math import log, exp
import numpy as np

class SimulatedData:
    def __init__(self, hr_ratio,
        average_death = 5, end_time = 15,
        num_features = 10, num_var = 2,
        treatment_group = False):
        """Factory class for producing simulated survival data.

        Current supports two forms of simulated data:
            Linear:
                Where risk is a linear combination of an observation's features
            Nonlinear (Gaussian):
                A gaussian combination of covariates

        Parameters
        ----------
        hr_ratio: int or float
            lambda_max hazard ratio.
        average_death: int or float
            average death time that is the mean of the Exponentional distribution.
        end_time: int or float
            censoring time that represents an 'end of study'. Any death 
            time greater than end_time will be censored.
        num_features: int
            size of observation vector. Default: 10.
        num_var: int
            number of varaibles simulated data depends on. Default: 2.
        treatment_group: bool
            True or False. Include an additional covariate representing a binary treatment group.
        """

        self.hr_ratio = hr_ratio
        self.end_time = end_time
        self.average_death = average_death
        self.treatment_group = treatment_group
        self.m = int(num_features) + int(treatment_group)
        self.num_var = num_var

    def _linear_H(self,x):
        """Calculates a linear combination of x's features.

        Coefficients are 1, 2, ..., self.num_var, 0,..0]

        Parameters
        ----------
        x: np.array
            (n,m) numpy array of observations

        Returns
        -------
        np.array
            risk: the calculated linear risk for a set of data x
        """
        # Make the coefficients [1,2,...,num_var,0,..0]
        b = np.zeros((self.m,))
        b[0:self.num_var] = range(1,self.num_var + 1)

        # Linear Combinations of Coefficients and Covariates
        risk = np.dot(x, b)
        return risk

    def _gaussian_H(self,x,
        c= 0.0, rad= 0.5):
        """Calculates the Gaussian function of a subset of x's features.

        Parameters
        ----------
        x: np.array
            (n, m) numpy array of observations.
        c: float
            offset of Gaussian function. Default: 0.0.
        r: float
            Gaussian scale parameter. Default: 0.5.

        Returns
        -------
        np.array
            risk: the calculated Gaussian risk for a set of data x
        """
        max_hr, min_hr = log(self.hr_ratio), log(1.0 / self.hr_ratio)

        # Z = ( (x_0 - c)^2 + (x_1 - c)^2 + ... + (x_{num_var} - c)^2)
        z = np.square((x - c))
        z = np.sum(z[:,0:self.num_var], axis = -1)

        # Compute Gaussian
        risk = max_hr * (np.exp(-(z) / (2 * rad ** 2)))
        return risk


    def generate_data(self, N,
        method = 'gaussian', seed = 1, censor_rate=0.5, gaussian_config = {}, censor_type = "log",
        **kwargs):
        """Generates a set of observations according to an exponentional Cox model.

        Parameters
        ----------
        N: int
            the number of observations.
        method: string
            the type of simulated data. 'linear' or 'gaussian'.
        guassian_config: dict
            dictionary of additional parameters for gaussian simulation.

        Returns
        -------
        dict
            dataset: a dictionary object with the following keys:
            'x' : (N,m) numpy array of observations.
            'e' : (N) numpy array of observed time events.
            't' : (N) numpy array of observed time intervals.
            'hr': (N) numpy array of observed true risk.

        Notes
        -----
        Peter C Austin. Generating survival times to simulate cox proportional
        hazards models with time-varying covariates. Statistics in medicine,
        31(29):3946-3958, 2012.
        """
        # Set random state
        np.random.seed(seed)

        # Patient Baseline information(N, m)
        data = np.random.uniform(low= -1, high= 1,
            size = (N,self.m))

        if self.treatment_group:
            data[:,-1] = np.squeeze(np.random.randint(0,2,(N,1)))
            print(data[:,-1])

        # Each patient has a uniform death probability
        p_death = self.average_death * np.ones((N,1))

        if method == 'linear':
            risk = self._linear_H(data)

        elif method == 'gaussian':
            risk = self._gaussian_H(data,**gaussian_config)

        # Center the hazard ratio so population dies at the same rate
        # independent of control group (makes the problem easier)
        risk = risk - np.mean(risk)

        # Generate time of death for each patient
        # currently exponential random variable
        death_time = np.zeros((N,1))
        for i in range(N):
            if self.treatment_group and data[i,-1] == 0:
                death_time[i] = np.random.exponential(p_death[i])
            else:
                death_time[i] = np.random.exponential(p_death[i]) / exp(risk[i])
        ### log censor
        censor_time = np.random.uniform(low= 1, high= 2,
            size = (N,1))
        if censor_type == "log":
            censor_time = np.log2(censor_time)
        elif censor_type == "exp":
            censor_time = np.exp(censor_time - 2)
        elif censor_type == "uniform":
            censor_time = censor_time -1
        censor_threshold = np.random.uniform(low= 0, high= 1,
            size = (N,1))

        censor_time[censor_threshold > censor_rate] = 1
        death_time = death_time * censor_time
        censoring = np.ones((N,1))
        censoring[censor_threshold <= censor_rate] = 0

        #


        dataset = {
            'x' : data.astype(np.float32),
            'e' : censoring.astype(np.int32),
            't' : death_time.astype(np.float32),
            'hr' : risk.astype(np.float32)
        }

        return dataset