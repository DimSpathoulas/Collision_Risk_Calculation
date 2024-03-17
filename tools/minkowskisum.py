import numpy as np


def sort_vertices(polygon):
    """Sorts vertices by polar angles.

    Args:
        polygon (list[list[float, float]]): list of polygon vertices

    Returns:
        list[list[float, float]]: list of polygon vertices sorted
    """
    cx, cy = polygon.mean(0)  # center of mass
    x, y = polygon.T
    angles = np.arctan2(y-cy, x-cx)
    indices = np.argsort(angles)
    return polygon[indices]


def crossprod(p1, p2):
    """Cross product of two vectors in 2R space.

    Args:
        p1 (list[float, float]): first vector
        p2 (list[float, float): second vector

    Returns:
        float: value of cross product
    """
    return p1[0] * p2[1] - p1[1] * p2[0]


def minkowskisum(pol1, pol2):
    """Calculate Minkowski sum of two convex polygons.

    Args:
        pol1 (np.ndarray[float, float]): first polygon
        pol2 (np.ndarray[float, float]): second polygon

    Returns:
        np.ndarray[np.ndarray[float, float]]: list of the Minkowski sum vertices
    """
    msum = []
    pol1 = sort_vertices(pol1)
    pol2 = sort_vertices(pol2)

    # sort vertices so that is starts with lowest y-value
    min1, min2 = np.argmin(pol1[:, 1]), np.argmin(pol2[:, 1])  # index of vertex with min y value
    pol1 = np.vstack((pol1[:min1], pol1[min1:]))
    pol2 = np.vstack((pol2[:min2], pol2[min2:]))

    i, j = 0, 0
    l1, l2 = len(pol1), len(pol2)
    # iterate through all the vertices
    while i < len(pol1) and j < len(pol2):
        msum.append(pol1[i % l1] + pol2[j % l2])
        cross = crossprod((pol1[(i+1) % l1] - pol1[i % l1]), pol2[(j+1) % l2] - pol2[j % l2])
        # using right-hand rule choose the vector with the lower polar angle and iterate this polygon's vertex
        if cross >= 0:
            i += 1
        if cross <= 0:
            j += 1

    return np.array(msum)
