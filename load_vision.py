"""Vision log handling module UNM 2014-06-25"""

__all__ = ['load_vision', 'load_vision_list']

from pyhdf.SD import SD, SDC
import numpy as np
from math import ceil
from scipy import interpolate
from os.path import exists


def load_vision(filename, var=None, T=1):
    """load_vision loads a Vision log file and
       returns its content in a dict.
    """

    assert exists(filename), 'Invalid filename.'

    f = SD(filename, SDC.READ)

    # New time axis
    end = ceil(f.select('ts_group_0').get()[-1])
    new_time = np.arange(0, end, T)

    # Initialize dict
    req_data = {'t': new_time}

    # Loop over variable list and loaded signals to search for matches
    if not var:
        req_data.update({key.split('.')[-1]: _select_interp(new_time, f, key)
                         for key in f.datasets().keys()
                         if not key.startswith('ts_')})
    elif isinstance(var, basestring):
        first_match = next((key for key in f.datasets().keys() if var in key),
                           None)
        req_data.update({var: _select_interp(new_time, f, first_match)})
    else:
        first_match = zip(var,
                          [next((key for key in f.datasets().keys()
                                 if sig in key), None)
                           for sig in var])
        req_data.update({sig: _select_interp(new_time, f, key)
                         for sig, key in first_match})

    f.end()

    return req_data


def _select_interp(new_time, f, key):
    """Handle the interpolation of data.
       Return an array of nans if key is not found.
    """

    if key in f.datasets().keys():
        data = f.select(key)
        Y = data.get()
        X = f.select(getattr(data, 'event')).get()
        data.endaccess()
        F = interpolate.interp1d(X, Y, kind='nearest', bounds_error=False)
        new_signal = F(new_time)
    else:
        new_signal = np.empty(new_time.shape)
        new_signal.fill(np.nan)

    return new_signal


# Vision list
def load_vision_list(files, var=None, T=1):
    """load_vision loads a list of Vision log files and
       appends their content in a returned dict.
    """

    # TODO This could possibly be improved using a generator or
    # iterator to avoid copies of data.
    dicts = [load_vision(file_name, var, T) for file_name in files if exists(file_name)]

    signals = {key: np.append(*[dct[key] for dct in dicts])
               for key in dicts[0].keys()}

    return signals
