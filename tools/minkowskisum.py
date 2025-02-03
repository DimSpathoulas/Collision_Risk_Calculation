import numpy as np

def sort_vertices(polygon):
    """Sorts vertices by polar angles.
    
    Args:
        polygon (np.ndarray[float, float]): list of polygon vertices
    
    Returns:
        np.ndarray[float, float]: list of polygon vertices sorted
    """
    cx, cy = polygon.mean(0)  # center of mass
    x, y = polygon.T
    angles = np.arctan2(y-cy, x-cx)
    indices = np.argsort(angles)
    return polygon[indices]

def crossprod(p1, p2):
    """Cross product of two vectors in 2D space.
    
    Args:
        p1 (np.ndarray[float, float]): first vector
        p2 (np.ndarray[float, float]): second vector
    
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
        np.ndarray[float, float]: list of the Minkowski sum vertices
    """
    # Sort vertices counterclockwise
    pol1 = sort_vertices(pol1)
    pol2 = sort_vertices(pol2)
    
    # Find starting points (lowest y-coordinate)
    min1 = np.argmin(pol1[:, 1])
    min2 = np.argmin(pol2[:, 1])
    
    # Reorder arrays to start with lowest points
    pol1 = np.roll(pol1, -min1, axis=0)
    pol2 = np.roll(pol2, -min2, axis=0)
    
    msum = []
    i, j = 0, 0
    l1, l2 = len(pol1), len(pol2)
    
    # Track visited vertex pairs to prevent duplicates
    visited = set()
    
    while i < l1 or j < l2:
        # Skip if we've already processed this vertex pair
        if (i % l1, j % l2) in visited:
            break
            
        visited.add((i % l1, j % l2))
        msum.append(pol1[i % l1] + pol2[j % l2])
        
        # Calculate cross product of edge vectors
        edge1 = pol1[(i + 1) % l1] - pol1[i % l1]
        edge2 = pol2[(j + 1) % l2] - pol2[j % l2]
        cross = crossprod(edge1, edge2)
        
        # Update indices based on cross product
        if cross >= 0:
            i += 1
        if cross <= 0:
            j += 1
    
    result = np.array(msum)
    
    # Remove any duplicate points that might be very close to each other
    result = np.unique(result.round(decimals=10), axis=0)
    
    return result
