from pyquaternion import Quaternion

import numpy as np
import collections

from shared.pyutils.tensorutils import UtilCartesianMatrix

# open3d is not compatible with PyTorch !
import open3d as o3d


def UtilDisplay3d(object_list):
  """
  Display 3d objects using open3d library
  :param object_list: List of tuples: (points, lines, colors)
    points: numpy[N, 3], float
    lines: numpy[M, 2], int OR None
    colors:
      None - black
      tuple of 3 colors - paint in this color
      string "axis" - paint by axis, X-R Y-G Z-B
      numpy[N, 3] or numpy[M, 3] - corresponding colors
  :return:
  """
  o3d_list = []
  for points, lines, colors in object_list:
    assert len(points.shape) == 2
    assert points.shape[1] == 3

    if lines is None:
      pcd = o3d.geometry.PointCloud()
    else:
      pcd = o3d.geometry.LineSet()
      assert len(lines.shape) == 2
      assert lines.shape[1] == 2
      assert lines.dtype == np.int
      pcd.lines = o3d.utility.Vector2iVector(lines)

    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is None:
      # Paint black
      pcd.paint_uniform_color([0., 0., 0.])
    elif isinstance(colors, tuple):
      color = np.array(list(colors)).astype(np.float) / 255
      pcd.paint_uniform_color(color)
    elif isinstance(colors, str) and (colors == "axis"):
      assert lines is None
      min_vals = np.min(points, axis=0)
      max_vals = np.max(points, axis=0)
      diff = (max_vals - min_vals).clip(0.000001)
      point_colors = (points - min_vals) / diff
      pcd.colors = o3d.utility.Vector3dVector(point_colors)
    else:
      if lines is None:
        assert points.shape == colors.shape
      else:
        assert lines.shape[0] == colors.shape[0]
      pcd.colors = o3d.utility.Vector3dVector(colors / 255.)

    o3d_list.append(pcd)

  o3d.draw_geometries(o3d_list)


def UtilCreateBoxGeometry(center, dimensions, rotation, color):
  vertices = UtilCartesianMatrix([-dimensions[0] / 2., dimensions[0] / 2.],
                                 [-dimensions[1] / 2., dimensions[1] / 2.],
                                 [-dimensions[2] / 2., dimensions[2] / 2.]).reshape(8, 3)
  lines = np.array([[0, 1], [0, 2], [0, 4],
                    [1, 3], [1, 5],
                    [2, 3], [2, 6],
                    [3, 7],
                    [4, 5], [4, 6],
                    [5, 7],
                    [6, 7]])
  rot_matrix_trans = Quaternion(rotation).rotation_matrix.T
  vertices = np.dot(vertices, rot_matrix_trans)
  vertices += center
  return (vertices, lines, color)

