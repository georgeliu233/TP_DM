import numpy as np

class RunningStat(object):
    '''
    Keeps track of first and second moments (mean and variance)
    of a streaming time series.
     Taken from https://github.com/joschu/modular_rl
     Math in http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape,(x.shape,self._M.shape)
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """
    def __init__(self, shape, center=True, scale=True, clip=None,gamma=None):
        assert shape is not None
        self.center = center
        self.scale = scale
        self.clip = clip
        self.shape = shape
        self.rs = RunningStat(self.shape)
        self.gamma=gamma
        if gamma:
            self.ret = np.zeros(shape)

        
        # self.prev_filter = prev_filter

    def __call__(self, x, **kwargs):
        # x = self.prev_filter(x, **kwargs)
        # print(x)
        if self.gamma:
            self.ret = self.ret * self.gamma + x
            self.rs.push(self.ret)
        else:
            self.rs.push(x)
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean
                diff = diff/(self.rs.std + 1e-8)
                x = diff + self.rs.mean
                # x = x/(self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        # self.prev_filter.reset()
        if self.gamma:
            self.ret = np.zeros_like(self.ret)
        self.rs = RunningStat(self.shape)

class Identity:
    '''
    A convenience class which simply implements __call__
    as the identity function
    '''
    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass
    
def test_r():
    r_filter = ZFilter(shape=(), center=False)

    r_list = []
    for r in range(10):
        rew = r_filter(r)
        print(r,rew)
    
    r_filter.reset()
    for r in range(10):
        rew = r_filter(r)
        print(r,rew)

if __name__=="__main__":
    test_r()
