from neural_net.evaluation import ConfusionMatrix, RegressionMetrics

import numpy as np
import unittest

TEST_MATRIX = np.array([[10, 2, 3], [6, 20, 1], [3, 5, 15]])

class TestConfusionMatrix(unittest.TestCase):

	def test_TP(self):
		cf_matrix = ConfusionMatrix(3)
		cf_matrix.matrix = TEST_MATRIX
		expected_TP = np.array([10, 20, 15])

		actual_TP = cf_matrix._TP()

		self.assertTrue(np.equal(actual_TP, expected_TP).all())

	def test_FP(self):
		cf_matrix = ConfusionMatrix(3)
		cf_matrix.matrix = TEST_MATRIX
		expected_FP = np.array([5, 7, 8])

		actual_FP = cf_matrix._FP()

		self.assertTrue(np.equal(actual_FP, expected_FP).all())

	def test_FN(self):
		cf_matrix = ConfusionMatrix(3)
		cf_matrix.matrix = TEST_MATRIX
		expected_FN = np.array([9, 7, 4])

		actual_FN = cf_matrix._FN()

		self.assertTrue(np.equal(actual_FN, expected_FN).all())

	def test_precision(self):
		cf_matrix = ConfusionMatrix(3)
		cf_matrix.matrix = TEST_MATRIX
		expected_precision = np.array([0.6666667, 0.740740741, 0.652173913])

		actual_precision = cf_matrix.precision()

		self.assertTrue(np.isclose(actual_precision, expected_precision).all())

	def test_recall(self):
		cf_matrix = ConfusionMatrix(3)
		cf_matrix.matrix = TEST_MATRIX

		expected_recall = np.array([0.526315789, 0.740740741, 0.789473684])

		actual_recall = cf_matrix.recall()

		self.assertTrue(np.isclose(actual_recall, expected_recall).all())

	def test_accuracy(self):
		cf_matrix = ConfusionMatrix(3)
		cf_matrix.matrix = TEST_MATRIX
		expected_acc = 0.692307692

		actual_acc = cf_matrix.accuracy()

		self.assertLess(abs(actual_acc - expected_acc), 10e-6)

	def test_F1_score(self):
		cf_matrix = ConfusionMatrix(3)
		cf_matrix.matrix = TEST_MATRIX
		expected_f1 = np.array([0.588235288, 0.740740741, 0.714285715])

		actual_f1 = cf_matrix.F1_score()

		self.assertTrue(np.isclose(actual_f1, expected_f1).all())

	def test_F1_macro(self):
		cf_matrix = ConfusionMatrix(3)
		cf_matrix.matrix = TEST_MATRIX
		expected_f1_macro = 0.681087248

		actual_f1_macro = cf_matrix.F1_macro()

		self.assertLess(abs(actual_f1_macro - expected_f1_macro), 10e-6)

	def test_F1_micro(self):
		cf_matrix = ConfusionMatrix(3)
		cf_matrix.matrix = TEST_MATRIX
		expected_f1_micro = 0.692307693

		actual_f1_micro = cf_matrix.F1_micro()

		self.assertLess(abs(actual_f1_micro - expected_f1_micro), 10e-6)

	def test_RMSE(self):
		pred = np.array([[100, 5600, 251]])
		Y = np.array([[150, 9000, 146]])
		expected_rmse = 3401.988389163

		actual_rmse = RegressionMetrics.rmse(pred, Y)

		self.assertLess(abs(actual_rmse - expected_rmse), 10e-6)