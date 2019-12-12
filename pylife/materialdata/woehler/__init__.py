import pandas as pd

@pd.api.extensions.register_series_accessor('woehler')
class WoehlerDataAccessor:
    def __init__(self, obj):
        self._validate(obj)
        self._obj = obj

    def _validate(self, obj):
        if not 'strength_inf' in obj.index or not 'strength_scatter' in obj.index:
            raise AttributeError('need at least "strengh_inf" and "strength_scatter"')
