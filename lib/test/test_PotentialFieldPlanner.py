import unittest
from lib.potentialFieldPlanner import PotentialFieldPlanner
import numpy as np
from lib.loadmap import loadmap
from copy import deepcopy
from numpy import pi

class TestPotentialFieldPlanner(unittest.TestCase):

    def setUp(self):
        return super().setUp()

    def test_attractive_force(self):
        """Test the attractive force calculation"""
        zeta = 30
        target = np.array([[2], [2], [2]])
        current = np.array([[1], [1], [1]])
        expected_force = np.array([[1/np.sqrt(3)], [1/np.sqrt(3)], [1/np.sqrt(3)]])
        force = PotentialFieldPlanner.attractive_force(target, current)
        np.testing.assert_array_almost_equal(force, expected_force)

    def test_attractive_force_2(self):
        """Test the attractive force calculation"""
        zeta = 30
        target = np.array([[1.1], [1.1], [1.1]])
        current = np.array([[1], [1], [1]])
        expected_force = np.array([[zeta*0.1], [zeta*0.1], [zeta*0.1]])
        force = PotentialFieldPlanner.attractive_force(target, current)
        np.testing.assert_array_almost_equal(force, expected_force)

    def test_repulsive_force(self):
        """Test the repulsive force calculation"""
        eta = 10

        obstacle = np.array([1, 1, 1, 2, 2, 2])
        current = np.array([[0.5], [1], [1]])
        expected_force = np.array([[-eta/(0.125)], [0], [0]])

        force = PotentialFieldPlanner.repulsive_force(obstacle, current)
        np.testing.assert_array_almost_equal(force, expected_force)

    def test_compute_forces(self):
        """Test the combined attractive and repulsive forces calculation"""
        target = np.random.rand(3, 9)
        current = np.random.rand(3, 9)
        obstacles = [np.array([0, 0, 0, 1, 1, 1])]

        forces = PotentialFieldPlanner.compute_forces(target, obstacles, current)

        self.assertEqual(forces.shape, (3, 9))
        self.assertTrue(np.all(forces != 0))

    def test_linear_jacobian_transpose(self):
        """Test the linear jacobian transpose"""
        q_in = np.array([1, 1, 1, 1, 1, 1, 1])
        joint_index = 3

        jacobian = PotentialFieldPlanner.linear_jacobian_transpose(q_in, joint_index)

        self.assertEqual(jacobian.shape, (3, joint_index))

    def test_linear_jacobian_transpose2(self):
        """Test the linear jacobian transpose"""
        q_in = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
        joint_index = 6

        jacobian = np.round(PotentialFieldPlanner.linear_jacobian_transpose(q_in, joint_index), 4)

        self.assertEqual(jacobian.shape, (3, joint_index))

    def test_compute_torques_1(self):
        """Test torque calculation from joint forces"""
        joint_forces = np.zeros(shape=(3,9))
        q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
        
        torques = PotentialFieldPlanner.compute_torques(joint_forces, q)
        
        self.assertEqual(torques.shape, (1, 9))
        self.assertTrue(np.all(torques == 0))

    def test_compute_torques_2(self):
        """Test torque calculation from joint forces"""
        joint_forces = np.zeros(shape=(3,9))
        joint_forces[1,3] = 10
        q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
        
        torques = PotentialFieldPlanner.compute_torques(joint_forces, q)
        expected_torques = np.array([[0.825, 0, 0.825, 0, 0, 0, 0, 0, 0]])
        
        self.assertEqual(torques.shape, (1, 9))
        np.testing.assert_array_almost_equal(torques, expected_torques)

    def test_q_distance(self):
        """Test the distance between two joint configurations"""
        target = np.array([1, 1, 1, 1, 1, 1, 1])
        current = np.array([2, 2, 2, 2, 2, 2, 2])

        distance = PotentialFieldPlanner.q_distance(target, current)
        self.assertEqual(distance, np.linalg.norm(target - current))

    def test_compute_gradient(self):
        """Test the gradient computation"""
        q = np.array([0, 0, 0, 0, 0, 0, 0])
        target = np.array([1, 1, 1, 1, 1, 1, 1])
        map_struct = loadmap("maps/map1.txt")
        map_struct.obstacles = [np.array([0, 0, 0, 1, 1, 1])]

        dq = PotentialFieldPlanner.compute_gradient(q, target, map_struct)

        self.assertEqual(dq.shape, (1, 7))
        self.assertTrue(np.any(dq != 0))

    def test_is_movement_valid(self):
        """Test if a movement from one configuration to another is valid"""
        q = np.array([0, 0, 0, 0, 0, 0, 0])
        q_new = np.array([1, 1, 1, 1, 1, 1, 1])
        obstacles = [np.array([0, 0, 0, 1, 1, 1])]

        self.assertFalse(PotentialFieldPlanner.is_movement_valid(q, q_new, obstacles))
    
    def test_is_movement_valid2(self):
        """Test if a movement from one configuration to another is valid"""
        q = np.array([0, 0, 0, 0, 0, 0, 0])
        q_new = np.array([1, 1, 1, 1, 1, 1, 1])
        obstacles = [np.array([0, 0, 0, 1, 1, 1])]

        self.assertFalse(PotentialFieldPlanner.is_movement_valid(q, q_new, obstacles))

    def test_is_movement_valid3(self):
        """Test if a movement from one configuration to another is valid"""
        q = np.array([0, 0, 0, 0, 0, 0, 0])
        q_new = np.array([1, 1, 1, 1, 1, 1, 1])
        obstacles = [np.array([2, 2, 2, 3, 3, 3])]

        self.assertTrue(PotentialFieldPlanner.is_movement_valid(q, q_new, obstacles))

    def test_is_movement_valid(self):
        """Test if a movement from one configuration to another is valid"""
        q = np.array([0, 0, 0, 0, 0, 0, 0])
        q_new = np.array([1, 1, 1, 1, 1, 1, 1])
        obstacles = [np.array([0, 0, 0, 1, 1, 1])]

        self.assertFalse(PotentialFieldPlanner.is_movement_valid(q, q_new, obstacles))

    def test_plan(self):
        """Test the full potential field planner"""
        start = np.array([0, 0, 0, 0, 0, 0, 0])
        goal = np.array([1, 1, 1, 1, 1, 1, 1])
        map_struct = loadmap("maps/map1.txt")
        map_struct.obstacles = [np.array([0, 0, 0, 1, 1, 1])]

        # Mocking the plan function to return a path
        q_path = self.planner.plan(map_struct, start, goal)

        self.assertIsInstance(q_path, np.ndarray)
        self.assertGreater(q_path.shape[0], 1)  
        self.assertTrue(np.allclose(q_path[-1], goal))


    def test_plan_complex(self):
        np.set_printoptions(suppress=True,precision=5)

        planner = PotentialFieldPlanner()

        # inputs 
        #map_struct = loadmap("maps/map1.txt")
        MyStruct = namedtuple("map", "obstacles")
        map_struct = MyStruct(obstacles = np.array([]))
        start = np.array([0,-1,0,-2,0,1.57,0])
        goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])

        # potential field planning
        q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))

        # show results
        for i in range(q_path.shape[0]):
            error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
            print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

        print("q path: ", q_path) 