import numpy as np


def test_parent_and_child_intercepts_are_structurally_coupled() -> None:
    # Two parent groups, each with two children and two observations per child.
    z = np.zeros((8, 6), dtype=np.float64)
    for row in range(8):
        z[row, row // 4] = 1.0
        z[row, 2 + row // 2] = 1.0
    crossproduct = z.T @ z
    assert crossproduct[0, 2] == 2.0
    assert crossproduct[0, 3] == 2.0
    assert crossproduct[1, 4] == 2.0
    assert crossproduct[1, 5] == 2.0
