import numpy as np

class BackgroundLike(object):

    def __init__(self,data, model):
        """
        
        :param data: 
        :param model: 
        """

        self._model = model

        self._free_parameters = self._model.free_parameters


        #TODO: the data object should return all the time bins that are valid... i.e. non-zero
        self._time_bins = None

        #TODO: extract the counts from the data object. should be same size as time bins
        self._counts = None





    def _evaluate_model(self):
        """
        
        loops over time bins and extracts the model flux and returns this array
        
        
        :return: 
        """

        pass



    def _set_free_parameters(self, new_parameters):
        """
        Set the free parameters to the new values
        :param new_parameters: 
        :return: 
        """

        for i, parameter in enumerate(self._free_parameters.itervalues()):


            parameter.value = new_parameters[i]


    def __call__(self, parameters):
        """
        
        :return: the poisson log likelihood
        """

        self._set_free_parameters(parameters)


        M = self._evaluate_model()
        M_fixed, tiny = self._fix_precision(M)

        # Replace negative values for the model (impossible in the Poisson context)
        # with zero

        negative_mask = (M < 0)
        if (len(negative_mask.nonzero()[0]) > 0):
            M[negative_mask] = 0.0

        # Poisson loglikelihood statistic (Cash) is:
        # L = Sum ( M_i - D_i * log(M_i))

        logM = self._evaluate_logM(M)

        # Evaluate v_i = D_i * log(M_i): if D_i = 0 then the product is zero
        # whatever value has log(M_i). Thus, initialize the whole vector v = {v_i}
        # to zero, then overwrite the elements corresponding to D_i > 0



        d_times_logM = self._counts * logM

        log_likelihood = np.sum(M_fixed - d_times_logM)

        return log_likelihood


    def _fix_precision(self, v):
      """
      Round extremely small number inside v to the smallest usable
      number of the type corresponding to v. This is to avoid warnings
      and errors like underflows or overflows in math operations.
    
    
      :param v: 
      :return: 
      """

      tiny = np.float64(np.finfo(v[0]).tiny)
      zero_mask = (np.abs(v) <= tiny)
      if (len(zero_mask.nonzero()[0]) > 0):
          v[zero_mask] = np.sign(v[zero_mask]) * tiny

      return v, tiny

    def _evaluate_logM(self, M):
        # Evaluate the logarithm with protection for negative or small
        # numbers, using a smooth linear extrapolation (better than just a sharp
        # cutoff)
        tiny = np.float64(np.finfo(M[0]).tiny)

        non_tiny_mask = (M > 2.0 * tiny)

        tink_mask = np.logical_not(non_tiny_mask)

        if (len(tink_mask.nonzero()[0]) > 0):
            logM = np.zeros(len(M))
            logM[tink_mask] = np.abs(M[tink_mask]) / tiny + np.log(tiny) - 1
            logM[non_tiny_mask] = np.log(M[non_tiny_mask])

        else:

            logM = np.log(M)

        return logM