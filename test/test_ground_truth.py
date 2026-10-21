"""Geometry, file parsing and online alignment regression tests."""

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'ground_truth_node.py'
SPEC = importlib.util.spec_from_file_location('ground_truth_node', SCRIPT)
gt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gt
SPEC.loader.exec_module(gt)
START = 1413394900000000000


def rotation_z(angle):
    return [0, 0, math.sin(angle/2), math.cos(angle/2)]


@pytest.fixture
def trajectory(tmp_path):
    filename = tmp_path / 'data.csv'
    rows = ['#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z']
    for i in range(3):
        x, y, z, w = rotation_z(i*math.pi/2)
        rows.append(f'{START+i*10000000},{i},0,0,{w},{x},{y},{z}')
    filename.write_text('\n'.join(rows))
    return gt.GroundTruthTrajectory(filename)


def test_interpolation_and_no_extrapolation(trajectory):
    pose = trajectory.sample(START+5000000)
    np.testing.assert_allclose(pose[:3, 3], [0.5, 0, 0])
    expected = gt.pose_matrix([0, 0, 0], rotation_z(math.pi/4))
    np.testing.assert_allclose(pose[:3, :3], expected[:3, :3], atol=1e-12)
    assert trajectory.sample(START-1) is None
    assert trajectory.sample(START+20000001) is None
    np.testing.assert_allclose(trajectory.sample(START+20000000)[:3, 3], [2, 0, 0])


def test_gap_is_not_interpolated(trajectory):
    trajectory.max_gap_ns = 5000000
    assert trajectory.sample(START+5000000) is None
    assert trajectory.sample(START) is not None


def test_tum_timestamp_precision_and_quaternion_order(tmp_path):
    filename = tmp_path / 'trajectory.txt'
    filename.write_text('1413394900.000000001 1 2 3 0 0 0 2\n'
                        '1413394900.010000001 2 3 4 0 0 0 1\n')
    trajectory = gt.GroundTruthTrajectory(filename)
    assert trajectory.stamps[0] == START+1
    np.testing.assert_allclose(trajectory.sample(START+1), gt.pose_matrix([1, 2, 3], [0, 0, 0, 1]))


def test_slerp_handles_quaternion_sign():
    q = np.array(rotation_z(math.pi/3))
    np.testing.assert_allclose(gt.slerp(q, -q, 0.5), q)


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('angle', [0, math.pi/3, math.pi-1e-8, math.pi])
def test_matrix_quaternion_round_trip(axis, angle):
    q = np.zeros(4)
    q[axis], q[3] = math.sin(angle/2), math.cos(angle/2)
    pose = gt.pose_matrix([0, 0, 0], q)
    recovered = gt.pose_matrix([0, 0, 0], gt.matrix_quaternion(pose[:3, :3]))
    np.testing.assert_allclose(recovered, pose, atol=1e-12)


def test_fixed_alignment_body_extrinsics_and_metrics(trajectory):
    body = gt.pose_matrix([0.2, 0.1, -0.3], [math.sin(0.2), 0, 0, math.cos(0.2)])
    alignment = gt.pose_matrix([3, 4, 5], rotation_z(0.7))
    comparison = gt.TrajectoryComparison(trajectory, body)
    first_estimate = alignment @ trajectory.sample(START) @ body
    first = comparison.update(START, first_estimate)
    assert first.position_error == pytest.approx(0, abs=1e-12)
    second_truth = alignment @ trajectory.sample(START+10000000) @ body
    second_estimate = second_truth.copy()
    second_estimate[:3, 3] += [0.3, 0.4, 0]
    second_estimate[:3, :3] = gt.pose_matrix([0, 0, 0], rotation_z(math.pi/18))[:3, :3] @ second_estimate[:3, :3]
    second = comparison.update(START+10000000, second_estimate)
    np.testing.assert_allclose(second.pose, second_truth, atol=1e-12)
    np.testing.assert_allclose(comparison.alignment, alignment, atol=1e-12)
    assert second.position_error == pytest.approx(0.5)
    assert second.orientation_error_deg == pytest.approx(10)
    assert second.position_rmse == pytest.approx(math.sqrt(0.25/2))
    assert comparison.update(START+10000000, second_estimate) is None
    assert comparison.count == 2


def test_reset_rewind_and_unmatched_samples(trajectory):
    comparison = gt.TrajectoryComparison(trajectory)
    identity = np.eye(4)
    assert comparison.update(START-1, identity) is None
    assert comparison.alignment is None and comparison.count == 0
    comparison.update(START+10000000, identity)
    generation = comparison.generation
    result = comparison.update(START, identity)
    assert comparison.generation == generation+1
    assert result.position_error == pytest.approx(0)
    assert comparison.count == 1
    comparison.reset()
    assert comparison.count == 0 and comparison.alignment is None
    shifted = gt.pose_matrix([20, 30, 40], rotation_z(1))
    assert comparison.update(START+10000000, shifted).position_error == pytest.approx(0)


@pytest.mark.parametrize('second', [
    f'{START},0,0,0,1,0,0,0',
    f'{START+1},0,0,0,0,0,0,0',
    f'{START+1},nan,0,0,1,0,0,0',
    'nan,0,0,0,1,0,0,0',
])
def test_rejects_invalid_data_with_line_number(tmp_path, second):
    filename = tmp_path / 'bad.csv'
    filename.write_text(f'{START},0,0,0,1,0,0,0\n{second}\n')
    with pytest.raises(ValueError, match='bad.csv:2:'):
        gt.GroundTruthTrajectory(filename)


def test_rejects_invalid_body_transform(trajectory):
    reflection = np.eye(4)
    reflection[0, 0] = -1
    with pytest.raises(ValueError, match='rigid transform'):
        gt.TrajectoryComparison(trajectory, reflection)
