from prerequisites import *

def test_equation():
    npa = np.arange(3)
    cpa = cp.arange(3)
    xpa = xp.array([npa, cpa], axis=0)
    # res = np.allclose(xpa, [npa, cpa])

def test_inequation():
    npa = np.arange(3)
    cpa = cp.arange(3)
    xpa = xp.array([npa, cpa], axis=0)
    less = (1 < xpa)
    print(less)
    less = (xpa < 2)
    print(less)

if __name__ == '__main__':
    # test_equation()
    test_inequation()