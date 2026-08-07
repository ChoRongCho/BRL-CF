import unittest

from baseline.attr_pomdp.scripts.rewards import TangentPWLCEntropy, negative_entropy


class RhoRewardTest(unittest.TestCase):
    def test_uniform_is_zero(self):
        self.assertAlmostEqual(negative_entropy([0.5, 0.5]), 0.0)

    def test_certain_is_log_cardinality(self):
        self.assertAlmostEqual(negative_entropy([1.0, 0.0]), 1.0)

    def test_tangent_is_lower_bound(self):
        probabilities = [0.8, 0.2]
        self.assertLessEqual(TangentPWLCEntropy()(probabilities), negative_entropy(probabilities) + 1e-9)


if __name__ == "__main__":
    unittest.main()
