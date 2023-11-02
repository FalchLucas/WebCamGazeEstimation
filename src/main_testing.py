"""
Only individual functions are tested, needs improvement
"""
import unittest
from gaze_tracking.tests.test_homtransform import TestHomTransform
from sfm.tests.test_sfm import TestSFM

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHomTransform)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestSFM)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromName('test_fitSTransG', TestHomTransform)
    unittest.TextTestRunner(verbosity=2).run(suite)