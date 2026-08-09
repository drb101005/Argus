import numpy as np

from app.recognizer.similarity import cosine_similarity
# recognizer/similarity.py

def test_identical_vectors():
    a = np.array([1.0, 2.0, 3.0])

    assert cosine_similarity(a, a) > 0.99


def test_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])

    assert cosine_similarity(a, b) < -0.99


def test_zero_vector_raises():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])

    try:
        cosine_similarity(a, b)
        assert False
    except ValueError:
        assert True