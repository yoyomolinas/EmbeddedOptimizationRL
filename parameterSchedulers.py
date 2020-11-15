class linearScheduler():
    """
    Implements:
    alpha = max(alpha0 * (c/(k*steps+c)) , minValue)
    """
    def __init__(self, initialValue = 1.0, minValue = 0.05, c = 1, k= 1):
        self.initVal = initialValue
        self.param = initialValue
        self.minVal =  minValue
        self.c = c
        self.k = k
        self.n = 0

    def update(self):
        self.param = max(self.initVal * (self.c/(self.k*self.n+self.c)), self.minVal)
        self.n += 1
