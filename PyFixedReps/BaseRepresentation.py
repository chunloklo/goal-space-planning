class BaseRepresentation:
    def encode(self, s, a = None):
        raise NotImplementedError()
    
    def features(self):
        raise NotImplementedError()
