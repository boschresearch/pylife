import pandas as pd

@pd.api.extensions.register_series_accessor('woehler')
class WoehlerDataAccessor:
    def __init__(self, obj, validator):
        self._validate(obj, validator)
        self._obj = obj

    def _validate(self, obj, validator):
        validator.fail_if_key_missing(obj, ['strength_inf', 'strength_scatter'])


