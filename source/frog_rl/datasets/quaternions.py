# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: BSD-3-Clause

"""Quaternion utilities for motion datasets.

Self-contained replacements for the ``pybullet_utils.transformations`` and
``pose3d`` / ``motion_util`` helpers used by the classic AMP motion loader.

Quaternions are stored in ``[x, y, z, w]`` order. All functions accept a single
quaternion of shape ``(4,)`` or a batch of quaternions of shape ``(N, 4)``.
"""

from __future__ import annotations

import torch


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    """Normalize a quaternion to unit length.

    Args:
        q: Quaternion(s) ``[x, y, z, w]`` of shape ``(4,)`` or ``(N, 4)``.

    Returns:
        The normalized quaternion(s).

    Raises:
        ValueError: If the input quaternion has near-zero norm.
    """
    q_norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    if torch.any(torch.isclose(q_norm, torch.zeros_like(q_norm))):
        raise ValueError(f"Quaternion may not be zero in quat_normalize: q = {q}")
    return q / q_norm


def standardize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Return a quaternion with ``w >= 0`` to remove the ``q = -q`` redundancy.

    Args:
        q: Quaternion(s) ``[x, y, z, w]`` of shape ``(4,)`` or ``(N, 4)``.

    Returns:
        A quaternion(s) with ``w >= 0``.
    """
    neg_mask = (q[..., -1:] < 0).expand_as(q)
    return torch.where(neg_mask, -q, q)


def quat_multiply(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions ``q0 * q1``.

    Args:
        q0: Quaternion(s) of shape ``(4,)`` or ``(N, 4)``.
        q1: Quaternion(s) of shape ``(4,)`` or ``(N, 4)``.

    Returns:
        The quaternion product ``q0 * q1``.
    """
    x0, y0, z0, w0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    return torch.stack(
        (
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        ),
        dim=-1,
    )


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """Conjugate (inverse) of a unit quaternion.

    Args:
        q: Quaternion(s) of shape ``(4,)`` or ``(N, 4)``.

    Returns:
        The conjugate quaternion(s).
    """
    return torch.stack((-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]), dim=-1)


def quat_from_axis_angle(axis: torch.Tensor, angle: float | torch.Tensor) -> torch.Tensor:
    """Return a quaternion that generates the given axis-angle rotation.

    Args:
        axis: Rotation axis of shape ``(3,)`` or ``(N, 3)``.
        angle: Rotation angle in radians.

    Returns:
        A unit quaternion(s) ``[x, y, z, w]``.
    """
    axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
    if torch.any(torch.isclose(axis_norm, torch.zeros_like(axis_norm))):
        raise ValueError(f"Axis vector may not have zero length: axis = {axis}")
    half_angle = angle * 0.5
    q = torch.zeros_like(axis, dtype=torch.float64)
    q[..., :3] = axis
    q[..., :3] *= torch.sin(half_angle) / axis_norm
    q[..., 3] = torch.cos(half_angle)
    return q


def quat_rotate_point(point: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Rotate a point by a quaternion using quaternion multiplication.

    Args:
        point: Point(s) of shape ``(3,)`` or ``(N, 3)``.
        quat: Quaternion(s) of shape ``(4,)`` or ``(N, 4)``.

    Returns:
        The rotated point(s).
    """
    q_point = torch.cat((point, torch.zeros_like(point[..., :1])), dim=-1)
    q_point_rotated = quat_multiply(quat_multiply(quat, q_point), quat_inverse(quat))
    return q_point_rotated[..., :3]


def quat_slerp(
    q0: torch.Tensor, q1: torch.Tensor, blend: float | torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions.

    Args:
        q0: Starting quaternion(s) of shape ``(4,)`` or ``(N, 4)``.
        q1: Ending quaternion(s) of shape ``(4,)`` or ``(N, 4)``.
        blend: Interpolation factor in ``[0, 1]``. A scalar or a tensor broadcastable
            to the batch dimension of the inputs.
        eps: Threshold below which linear interpolation is used to avoid division
            by zero at ``sin(theta) ~ 0``.

    Returns:
        The interpolated quaternion(s).
    """
    single = q0.dim() == 1
    if single:
        q0, q1 = q0.unsqueeze(0), q1.unsqueeze(0)

    # Convert the blend factor to the appropriate shape/dtype
    if not torch.is_tensor(blend):
        b = torch.full_like(q0[..., :1], float(blend))
    else:
        b = blend.to(q0.dtype).to(q0.device)
        if b.dim() == 0:
            b = b.reshape(1)
        if b.dim() == 1:
            b = b.unsqueeze(-1)

    # Take the shortest path between the two quaternions
    cos_half_theta = (q0 * q1).sum(dim=-1, keepdim=True)
    neg_mask = cos_half_theta < 0.0
    q1 = torch.where(neg_mask, -q1, q1)
    cos_half_theta = torch.where(neg_mask, -cos_half_theta, cos_half_theta)
    cos_half_theta = cos_half_theta.clamp(-1.0, 1.0)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta.square())

    # For near-zero angles use linear interpolation to avoid division by zero
    near_zero = sin_half_theta < eps
    w0 = torch.where(near_zero, 1.0 - b, torch.sin((1.0 - b) * half_theta) / (sin_half_theta + eps))
    w1 = torch.where(near_zero, b, torch.sin(b * half_theta) / (sin_half_theta + eps))

    out = w0 * q0 + w1 * q1
    return out.squeeze(0) if single else out
