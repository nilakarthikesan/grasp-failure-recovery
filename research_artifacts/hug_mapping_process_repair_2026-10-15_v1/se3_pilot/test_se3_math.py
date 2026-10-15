"""Manufactured controls; no saved cases, physics, fitting or new grasp data."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from se3_math import (decode_state,skew,left_jacobian_so3,left_compose_root_quaternion,
    rotation_point_jacobian,rotation_bounds_scaled,ROTATION_SCALE_RAD,ROTATION_COMPONENT_LIMIT_RAD)

def R(q):return Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix()

class Controls(unittest.TestCase):
    def test_zero_quaternion_bytes(self):
        q=np.r_[np.cos(.2),np.sin(.2),-0.,0.]
        self.assertEqual(left_compose_root_quaternion(q,np.zeros(3)).tobytes(),q.tobytes())
    def test_left_world_composition(self):
        q=np.r_[np.cos(.35),0.,np.sin(.35),0.];w=np.array([.12,-.07,.19])
        expected=Rotation.from_rotvec(w).as_matrix()@R(q)
        np.testing.assert_allclose(R(left_compose_root_quaternion(q,w)),expected,atol=5e-16,rtol=0)
        self.assertGreater(np.linalg.norm(expected-R(q)@Rotation.from_rotvec(w).as_matrix()),.01)
    def test_root_quaternion_unit_and_rotation(self):
        q=np.array([1.,0,0,0]);w=np.array([.2,-.15,.1]);actual=left_compose_root_quaternion(q,w)
        self.assertAlmostEqual(np.linalg.norm(actual),1.,places=15)
        np.testing.assert_allclose(R(actual),Rotation.from_rotvec(w).as_matrix(),atol=5e-16,rtol=0)
    def test_left_jacobian_zero(self):
        np.testing.assert_array_equal(left_jacobian_so3(np.zeros(3)),np.eye(3))
    def test_left_jacobian_small_angle(self):
        w=np.array([1e-9,-2e-9,3e-9]);expected=np.eye(3)+.5*skew(w)+skew(w)@skew(w)/6.
        np.testing.assert_allclose(left_jacobian_so3(w),expected,atol=1e-25,rtol=0)
    def test_point_jacobian_zero(self):self.check_point_jac(np.zeros(3))
    def test_point_jacobian_finite_rotation(self):self.check_point_jac(np.array([.18,-.13,.21]))
    def test_point_jacobian_near_corner(self):self.check_point_jac(np.array([.25,-.25,.25]))
    def check_point_jac(self,w):
        v=np.array([.031,-.074,.117]);point=Rotation.from_rotvec(w).apply(v)
        expected=rotation_point_jacobian(point,w);h=1e-5;actual=np.zeros((3,3))
        for j in range(3):
            d=np.eye(3)[j]*h*ROTATION_SCALE_RAD
            actual[:,j]=(Rotation.from_rotvec(w+d).apply(v)-Rotation.from_rotvec(w-d).apply(v))/(2*h)
        np.testing.assert_allclose(actual,expected,atol=3e-12,rtol=0)
    def test_decode_preserves_old_coordinates(self):
        x=np.arange(30,dtype=float)/10;q0=np.arange(24)/100
        q,t,w=decode_state(x,q0);np.testing.assert_array_equal(q,q0+.025*x[:24]);np.testing.assert_array_equal(t,.005*x[24:27]);np.testing.assert_array_equal(w,.025*x[27:])
    def test_rotation_bounds_are_component_bounds(self):
        b=rotation_bounds_scaled();self.assertTrue(np.all(b[:,1]*.025<=ROTATION_COMPONENT_LIMIT_RAD));self.assertTrue(np.all(b[:,0]*.025>=-ROTATION_COMPONENT_LIMIT_RAD))
        self.assertGreater(np.linalg.norm(b[:,1]*.025),ROTATION_COMPONENT_LIMIT_RAD)
    def test_bad_state_refused(self):
        for x in [np.zeros(27),np.full(30,np.nan)]:
            with self.assertRaises(ValueError):decode_state(x,np.zeros(24))
    def test_bad_quaternion_refused(self):
        with self.assertRaises(ValueError):left_compose_root_quaternion(np.array([2.,0,0,0]),np.zeros(3))
    def test_palm_compensated_rigid_motion_identity(self):
        palm_local=np.array([.03,.02,.09]);tip_local=np.array([.01,.04,.2]);anchor=np.array([.2,-.3,.4]);shift=np.array([.001,.002,-.003])
        R0=Rotation.from_rotvec([.3,.5,.2]).as_matrix();D=Rotation.from_rotvec([-.1,.2,-.1]).as_matrix();rot=D@R0
        root=anchor+shift-rot@palm_local;tip=root+rot@tip_local
        np.testing.assert_allclose(tip,anchor+shift+D@(R0@(tip_local-palm_local)),atol=1e-16,rtol=0)

if __name__=='__main__':unittest.main(verbosity=2)
