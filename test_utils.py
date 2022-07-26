import numpy as np
from utils import q_to_the, the_to_q

def test_q_to_the_1():
    q = np.array([0, 0, 1, 1, 0])
    the = np.array([0, 0, 1, 1, 0])
    out = q_to_the(q=q)
    assert np.all(np.equal(the, out))

def test_q_to_the_2():
    q = np.array([[1, 0, 1, 1, 0],
                       [1, 0, 1, 1, 0],
                       [0, 0, 0, 0, 0]])
    the = np.array([1, 1, 0, 1, 0, 1, 1, 0])
    out = q_to_the(q=q)
    assert np.all(np.equal(the, out))

def test_the_to_q_1():
    q = np.array([0, 0, 1, 1, 0])
    the = np.array([0, 0, 1, 1, 0])
    out = the_to_q(the=the, r=q.shape)
    assert np.all(np.equal(q, out))

def test_the_to_q_2():
    q = np.array([[1, 0, 1, 1, 0],
                  [1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0]])
    the = np.array([1, 1, 0, 1, 0, 1, 1, 0])
    out = the_to_q(the=the, r=q.shape)
    assert np.all(np.equal(q, out))

if __name__ == '__main__':
    test_q_to_the_1()
    test_q_to_the_2()
    test_the_to_q_1()
    test_the_to_q_2()
    print("All testing cases are passed!")