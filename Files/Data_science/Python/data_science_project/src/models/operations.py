import numpy as np

class Operations():
    def calculate_average(number_list:list):
        """
        Calculates the average of a list of numbers.

        :param number_list: A list of numbers.
        :type number_list: list of float
        :return: The average of the numbers.
        :rtype: float
        """

        if not isinstance(number_list, list):
            raise TypeError(f'Wrong type: {type(number_list)}, instead of list')
        else:
            return np.nanmean(number_list)