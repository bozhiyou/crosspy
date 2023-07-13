from prerequisites import *

def test_arithmetics():
    a = xp.array([np.arange(3), cp.arange(3)], axis=0)
    c = a + a
    print(c)

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
    test_arithmetics()
    # test_equation()
    test_inequation()
    # TODO sum
    # TODO argmin