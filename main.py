import numpy as np
import open3d as o3d
from numba import njit, jit, vectorize, float64



def main():
    pcd = o3d.io.read_point_cloud('./data/bun_zipper.ply')

    pcd = add_noise(pcd, 0.006)
    #o3d.visualization.draw_geometries([pcd])
    # filtering multiple times will reduce the noise significantly
    # but may cause the points distribute unevenly on the surface.
    pcd = guided_filter(pcd, 0.01, 0.25)
    pcd = guided_filter(pcd, 0.01, 0.25)
    # guided_filter(pcd, 0.01, 0.01)

    #o3d.visualization.draw_geometries([pcd])
@vectorize([float64(float64, float64, float64, float64, float64, float64)])
def np_filter(k, idx, epsilon, points, points_copy, i):
    neighbors = points[idx, :]
    mean = np.mean(neighbors)
    cov = np.cov(neighbors.T)
    e = np.linalg.inv(cov + epsilon * np.eye(3))

    A = cov @ e
    b = mean - A @ mean
    points_copy[i] = A @ points[i] + b
    return points_copy


def guided_filter(pcd, radius, epsilon):
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    points_copy = np.array(pcd.points)
    points = np.asarray(pcd.points)
    num_points = len(pcd.points)

    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        if k < 3:
            continue
        points_copy = np_filter(k, idx, epsilon, points, points_copy, i)

    pcd.points = o3d.utility.Vector3dVector(points_copy)
    return pcd 


def add_noise(pcd, sigma):
    points = np.asarray(pcd.points)
    noise = sigma * np.random.randn(points.shape[0], points.shape[1])
    points += noise

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


if __name__ == '__main__':
    main()
