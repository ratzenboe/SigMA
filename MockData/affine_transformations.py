from scipy.spatial.transform import Rotation
import numpy as np


def rotate_points_2d(points, angle=90):
    """Rotations of given data points around their mean in 3D

    Example: random rotation
    points_rotated = rotate_points_2d(points, angle=np.random.random()*360)
    """
    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    # Center points
    mean_points = np.mean(points, axis=0)
    points_centered = points - mean_points
    # Rotate
    points_rotated = points_centered @ R.T
    # Transform back
    points_transformed_final = points_rotated + mean_points
    return points_transformed_final


def rotate_points_3d(points, vector=np.array([1, 0, 0]), angle=90):
    """Rotations of given data points around their mean in 3D

    Example: random rotation
    vector = np.random.normal(loc=0.0, scale=1.0, size=3)
    angle = np.random.random()*360
    points_rotated = rotate_points_3d(points, vector=vector, angle=angle)
    """
    theta = np.deg2rad(angle)
    axis = vector / np.linalg.norm(vector)  # normalize the rotation vector first
    rot = Rotation.from_rotvec(theta * axis)
    # Center points
    mean_points = np.mean(points, axis=0)
    points_centered = points - mean_points
    # Rotate
    points_rotated = rot.apply(points_centered)
    # Transform back
    points_transformed_final = points_rotated + mean_points
    return points_transformed_final


def rotate_points_Nd(points, v1, v2, angle=90):
    """Rotations of given data points around their mean in N dimensions
    v1, v2 span a plane in which the rotation is performed

    Example: random rotation
    vector_1 = np.random.normal(loc=0.0, scale=1.0, size=5)
    vector_2 = np.random.normal(loc=0.0, scale=1.0, size=5)
    angle = np.random.random()*360
    points_rotated = rotate_points_Nd(points, v1=vector_1, v2=vector_2, angle=angle)
    """
    theta = np.deg2rad(angle)
    # Gram-Schmidt orthogonalization
    n1 = v1 / np.linalg.norm(v1)
    v2 = v2 - np.dot(n1, v2) * n1
    n2 = v2 / np.linalg.norm(v2)
    # N dimensional identity matrix
    I = np.identity(v1.size)
    # Rotation matrix
    R = I + (np.outer(n2, n1) - np.outer(n1, n2)) * np.sin(theta) + (np.outer(n1, n1) + np.outer(n2, n2)) * (np.cos(theta) - 1)
    # Center points
    mean_points = np.mean(points, axis=0)
    points_centered = points - mean_points
    # Rotate
    points_rotated = points_centered @ R.T
    # Transform back
    points_transformed_final = points_rotated + mean_points
    return points_transformed_final


def translate_points(points, target_mean):
    mean_points = np.mean(points, axis=0)
    # Translation direction is difference vector (Spitze minus Schaft)
    diff_vector = target_mean - mean_points
    translated_points = points + diff_vector
    return translated_points
