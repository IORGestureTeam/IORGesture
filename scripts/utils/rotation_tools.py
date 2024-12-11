import functools
from typing import Optional

import torch
import torch.nn.functional as F
from enum import Enum

class RotationType(Enum):
    ROT_6D = 0,
    AXIS_ANGLE = 1


"""
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""
def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix):
    return matrix_to_quaternion_v2(matrix)
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    """


def matrix_to_quaternion_v2(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)
def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def standardize_quaternion(quaternions):
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a, b):
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a, b):
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_invert(quaternion):
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    return quaternion * quaternion.new_tensor([1, -1, -1, -1])


def quaternion_apply(quaternion, point):
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def matrix_to_axis_angle_v2(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion_v2(matrix))


def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def optimize_rotation_matrices(R):
    """
    优化形状为[B, 3, 3]的旋转矩阵R使用SVD。
    参数:
        R (torch.Tensor): 待优化的旋转矩阵，形状为[B, 3, 3]。
    返回:
        torch.Tensor: 优化后的旋转矩阵，形状与输入相同。
    """
    # 确保R是浮点数类型
    R = R.float()
    
    # 初始化优化后的旋转矩阵
    R_optimized = torch.empty_like(R)
    
    # 遍历批次中的每个3x3子矩阵并优化
    for i in range(R.shape[0]):
        R_sub = R[i]
        
        # 对子矩阵进行SVD分解
        U, S, V = torch.svd(R_sub)
        
        # 构造最佳旋转矩阵UV'
        R_optimized[i] = U @ V.t()
    
    return R_optimized
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

def rotation_angle_difference(R1, R2):

    # 计算相对旋转 R_rel = R1^T * R2
    R_rel = torch.matmul(R1.transpose(-2, -1), R2)

    # 计算旋转矩阵的迹（trace），即对角线元素之和
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]

    #print(f"trace range: {trace.min()} ~ {trace.max()}")
    # 使用迹来计算旋转角度
    # 旋转角度theta可以通过acos((trace-1)/2)计算得到
    # 注意：这里的-1和2是欧拉角转换的常数项
    theta = torch.acos((torch.clamp(trace, max=3.0) - 1) / 2)

    print(f"nan theta: {theta.isnan().sum()}")

    # 由于acos的结果在[0, pi]范围内，我们可能需要调整它以获取最短的角度差
    # 如果trace < -1，则由于浮点数精度问题，acos将返回NaN。我们可以安全地假设这是180度
    theta[trace < -1 + 1e-5] = torch.tensor(torch.pi).to(theta.device)

    # 转换为度数（如果需要）
    theta_degrees = torch.mul(theta, 180.0 / torch.pi)

    return theta_degrees
def axis_angle_from_vector(v2):
    device = v2.device
    v1 = torch.zeros_like(v2) + torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    dot = torch.sum(v1 * v2, dim=-1, keepdim=True)
    axis = torch.zeros_like(v2)
    angle = torch.zeros([v2.shape[0], 1], dtype=torch.float32, device=device)
    # 计算特殊情况
    close_to_one = torch.isclose(dot, torch.tensor([1.0], device=device), atol=1e-6).squeeze(-1)
    close_to_minus_one = torch.isclose(dot, torch.tensor([-1.0], device=device), atol=1e-6).squeeze(-1)

    # 如果向量几乎相同
    axis[close_to_one] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    angle[close_to_one] = torch.tensor([0.0], dtype=torch.float32, device=device)

    # 如果向量几乎相反
    axis[close_to_minus_one] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    angle[close_to_minus_one] = torch.tensor(torch.pi, dtype=torch.float32, device=device)

    # 计算一般情况
    mask = ~(close_to_one | close_to_minus_one)
    cross = torch.cross(v1[mask], v2[mask], dim=-1)
    cross = cross / cross.norm(dim=-1, keepdim=True)
    axis[mask] = cross
    angle[mask] = torch.acos(dot[mask])
    #print(f"axis: {axis} angle: {angle}")
    return axis, angle
def rotation_matrix_around_axis(axis, angle):
    batch_size = axis.shape[0]
    device = axis.device
    c, s = torch.cos(angle), torch.sin(angle)
    I = torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, 1, 1)
    K = torch.zeros([batch_size, 3, 3], dtype=torch.float32, device=device)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    R = I + s.unsqueeze(-1).repeat(1, 3, 3) * K + (1 - c).unsqueeze(-1).repeat(1, 3, 3) * torch.matmul(K, K)
    #print(f"R: {R}")
    return R
def rotation_matrix_from_vector(v):
    axis, angle = axis_angle_from_vector(v)
    return rotation_matrix_around_axis(axis, angle)
def rotation_matrix_to_euler_angles(R):
    device = R.device
    theta_y = torch.zeros([R.shape[0]], dtype=torch.float32, device=device)
    theta_x = torch.zeros_like(theta_y)
    theta_z = torch.zeros_like(theta_y)
    # 提取旋转矩阵的元素
    mask = R[..., 2, 0].abs() != 1
    theta_y[mask] = -torch.asin(R[..., 2, 0][mask])
    cy = torch.cos(theta_y[mask])
    theta_x[mask] = torch.atan2(R[..., 2, 1][mask] / cy, R[..., 2, 2][mask] / cy)
    theta_z[mask] = torch.atan2(R[..., 1, 0][mask] / cy, R[..., 0, 0][mask] / cy)
    theta_z[~mask] = 0
    maskP = R[..., 2, 0] == 1
    maskM = R[..., 2, 0] == -1
    theta_y[maskP] = -torch.pi/2
    theta_y[maskM] = torch.pi/2
    theta_z[maskP] = torch.atan2(-R[..., 0, 1][maskP], -R[..., 0, 2][maskP])
    theta_z[maskM] = torch.atan2(R[..., 0, 1][maskM], R[..., 0, 2][maskM])

    # 将弧度转换为度
    theta_z = theta_z * (180.0 / torch.pi)
    theta_y = theta_y * (180.0 / torch.pi)
    theta_x = theta_x * (180.0 / torch.pi)

    # 返回结果，形状为[N, 3]
    return torch.stack((theta_z, theta_y, theta_x), dim=-1)
def euler_angles_to_rotation_matrix(theta):
    
    print(f"<euler_angles_to_rotation_matrix>nan theta: {theta.isnan().sum()}")
    # 确保theta是torch.Tensor类型，并且形状为(..., 3)
    theta = theta.type(torch.float)  # 确保是浮点数类型
    
    # 分离欧拉角（假设是Z-Y-X顺序）
    z, y, x = theta[..., 0], theta[..., 1], theta[..., 2]
    
    # 转换为弧度（如果输入已经是弧度，则这一步可以省略）
    z, y, x = z.mul(torch.pi / 180.0), y.mul(torch.pi / 180.0), x.mul(torch.pi / 180.0)
    
    # 计算旋转矩阵的组成部分
    cz = torch.cos(z)
    sz = torch.sin(z)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cx = torch.cos(x)
    sx = torch.sin(x)
    # 完整的旋转矩阵（Z-Y-X顺序）
    R = torch.stack([
        cy*cz,             cz*sy*sx - sz*cx,  cz*sy*cx + sz*sx,
        cy*sz,             sz*sy*sx + cz*cx,  sz*sy*cx - cz*sx,
        -sy,                cy*sx,              cy*cx
    ], dim=-1).reshape(*theta.shape[:-1], 3, 3)
    
    return R


def mirror_arm_bone(pose, start_id):
    return torch.cat((pose[..., start_id + 0:start_id + 1],
                      pose[..., start_id + 1:start_id + 2] * -1,
                      pose[..., start_id + 2:start_id + 3] * -1), dim=-1)
def arm_similarity(pose):
    pose_copy = pose.clone() # prevent torch original data overwrite
    def quat_similarity(q1, q2):
        return (torch.sum(q1 * q2, dim=-1) + 1) / 2 # 0 ~ 1
    arm1_1 = euler_to_quaternion(pose_copy[..., 9:12])
    arm1_2 = euler_to_quaternion(pose_copy[..., 12:15])
    arm1_3 = euler_to_quaternion(pose_copy[..., 15:18])
    arm2_1 = euler_to_quaternion(mirror_arm_bone(pose_copy, 75))
    arm2_2 = euler_to_quaternion(mirror_arm_bone(pose_copy, 78))
    arm2_3 = euler_to_quaternion(mirror_arm_bone(pose_copy, 81))
    #sum_euler_angle_diff = (pose_copy[..., 9:12] - mirror_arm_bone(pose_copy, 75)).abs() + (pose_copy[..., 12:15] - mirror_arm_bone(pose_copy, 78)).abs() + (pose_copy[..., 15:18] - mirror_arm_bone(pose_copy, 81)).abs()
    avg_similarity = (quat_similarity(arm1_1, arm2_1) + quat_similarity(arm1_2, arm2_2) + quat_similarity(arm1_3, arm2_3)) / 3
    return avg_similarity # for each batch, frame

def euler_to_quaternion(eulers, order='zyx'):
    """
    将欧拉角转换为四元数

    参数:
        eulers (torch.Tensor): 形状为 (..., 3) 的张量，包含N个欧拉角，每个欧拉角有3个分量
        order (str): 欧拉角的旋转顺序，例如 'zyx' 或 'xyz'

    返回:
        quats (torch.Tensor): 形状为 (..., 4) 的张量，包含N个四元数
    """
    
    eulers = eulers / 180 * torch.pi
    cy = torch.cos(eulers[..., 2] / 2.0)
    sy = torch.sin(eulers[..., 2] / 2.0)
    cp = torch.cos(eulers[..., 1] / 2.0)
    sp = torch.sin(eulers[..., 1] / 2.0)
    cr = torch.cos(eulers[..., 0] / 2.0)
    sr = torch.sin(eulers[..., 0] / 2.0)

    if order == 'zyx':
        quats = torch.stack((
            cr * cp * cy - sr * sp * sy,
            sr * cp * cy + cr * sp * sy,
            cr * sp * cy - sr * cp * sy,
            cr * cp * sy + sr * sp * cy
        ), dim=-1)
    elif order == 'xyz':
        # 类似地，你可以为其他顺序实现代码
        pass
    else:
        raise ValueError(f"Unsupported order: {order}")

    # 确保四元数规范化
    quats = quat_stabilize(quats)
    return quats
def quaternion_to_euler_xyz(quaternion):
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]

    # Compute roll (rotation around X-axis)
    t0 = +2.0 * (w * x - y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)
    #print(f"t0 {t0} a {w * x} b {y * z}")
    #print(f"t1 {t1} a {x * x} b {y * y}")

    # Compute pitch (rotation around Y-axis)
    t2 = +2.0 * (w * y + z * x)
    pitch = torch.asin(torch.clamp(t2, min=-1.0, max=1.0))

    # Compute yaw (rotation around Z-axis)
    t3 = +2.0 * (w * z - x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)
    #print(f"t3 {t3} a {w * z} b {x * y}")
    #print(f"t4 {t4} a {y * y} b {z * z}")

    return torch.stack((roll, pitch, yaw), dim=-1) * 180 / torch.pi
# Quaterion.py cannot support tensor
def quat_stabilize(quat, eps=1e-2):
    stable_quat = quat.clone()
    mask = quat[..., 0] < 0
    stable_quat[mask] *= -1
    norm = stable_quat.norm(dim=-1, keepdim=True)
    normalized_quat = stable_quat / norm
    return normalized_quat
'''
def quaternion_multiply(quat1, quat2):
    w0, x0, y0, z0 = quat1[..., 0], quat1[..., 1], quat1[..., 2], quat1[..., 3]
    w1, x1, y1, z1 = quat2[..., 0], quat2[..., 1], quat2[..., 2], quat2[..., 3]
    
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    
    return quat_stabilize(torch.stack([w, x, y, z], dim=-1))
'''
def quat_rel_rot(q1, q2):
    # 计算 q1 的逆四元数
    q1_inv = torch.stack([q1[..., 0], -q1[..., 1], -q1[..., 2], -q1[..., 3]], dim=-1)
    # 计算相对旋转 q1_inv * q2
    return quaternion_multiply(q1_inv, q2)