import numpy as np
import scipy.optimize as optimize
import pandas as pd


def Minimizer(self, likelihood):


    self._likelihood = likelihood


    def fit(self, n_interations = 100):

        # first do the linear fit for normalizations

        #optimize.minimize(self._likelihood)

        # now free all the needed parameters
        # and fit for them from the current starting positions



        # save the fit results and errors

        self._save()



        # display the results

        self.display()

    def _save(self):

        pass


    def display(self):
        """
        display the results using pandas series or dataframe
        
        
        :param self: 
        :return: 
        """

        pass