

class FeatureSet():

    def __init__(self):
        self._feature_lines = []
        self._y_feature_lines = []
        self._lookback = None
        self._normalizations = []

    def add_feature(self, feature_line):
        self._feature_lines.append(feature_line)

    def add_y_feature(self, feature_line):
        self._y_feature_lines.append(feature_line)

    def set_lookback(self, lookback):
        self._lookback = lookback

    def build(self, 
              start_date: datetime, 
              end_date: datetime, 
              target_type: str = 'next', 
              target_n: int = 1):
        '''
        Builds the features and target values for training of a neural
        network.
        '''
        if self._lookback is None:
            raise ValueError('Lookback must be set before building the feature set.')
        return self._build_X(), self._build_y(target_type, target_n)


    def _build_X(self):
        '''
        Builds the X matrix using the features and lookback passed into an instance of
        this class.

        Applies normalization if any normalization is set.

        Returns the X matrix as PyTorch tensor.
        #NOTE: Later on this might be abstracted such that
        it is possible to build an X matrix for different
        backend neural network frameworks.
        '''
        pass

    def _build_y(self, target_type: str = 'next', target_n: int = 1):
        '''
        Builds the y vector using the y features passed into an instance of this class.

        Params:
            target_type: str = 'next'
                The type of feature to use for the y vector.
                'next': the target value is the next value of the y feature.
                'perc': the target value is the percentage change of the next value 
                of the y feature.
                'direction': the target value is the direction of the next value of 
                the y feature (1 or -1).
            target_n: int = 1
                The number of steps to look ahead for the target value.
            For instance, if target_n is 1, the target value is the next value of the
            y feature.
            If target_n is 2, the target value is the value of the y feature two steps
            ahead.

        Returns the y vector as PyTorch tensor.
        #NOTE: Later on this might be abstracted such that
        it is possible to build a y vector for different
        backend neural network frameworks.
        '''
        y_features = self.build_y()
        if y_features is None:
            return self._build_y(target_type, target_n)
        #TODO: Implement the default logic.


    def build_y(self):
        '''
        This class can be overwritten to build the y vector for more complex targets.
        
        This class is called internally by the build method.
        It has to return a PyTorch tensor.
        '''
        pass

    def from_strategy(self, strategy):
        '''
        Extracts a features set from a strategy.
        NOTE: It might be better to implement this method
        in strategy as a `strat.to_feature_set()` method.
        '''
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def _apply_normalizations(self):
        pass