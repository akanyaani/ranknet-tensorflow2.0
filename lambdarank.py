from ranknet import *


class LTRModelLambdaRank(LTRModelRanknet):
	def __init__(self, activation=tf.nn.relu,
				 learning_rate=1e-3,
				 sigma=1.0,
				 dr_rate=0.25,
				 grad_clip=True,
				 clip_value=1.0):
		super(LTRModelLambdaRank, self).__init__(
			activation=activation,
			learning_rate=learning_rate,
			sigma=sigma,
			dr_rate=dr_rate,
			grad_clip=grad_clip,
			clip_value=clip_value
		)

		print("LambdaRank Model Initiated...............")

	def _get_lambdas(self, pred_score, labels):
		"""https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
		As explained in equation 3
		(1/2(1−Sij)−1/1+eσ(si−sj))"""
		with tf.name_scope("lambdas"):
			batch_size = tf.shape(labels)[0]

			index = tf.reshape(tf.range(1.0, tf.cast(batch_size,
													 dtype=tf.float32) + 1),
							   tf.shape(labels))

			sorted_labels = tf.sort(labels,
									direction="DESCENDING",
									axis=0)

			diff_matrix = labels - tf.transpose(labels)
			label_diff_matrix = tf.maximum(tf.minimum(1., diff_matrix), -1.)
			# print(label_diff_matrix)
			pred_diff_matrix = pred_score - tf.transpose(pred_score)
			# print(pred_diff_matrix)
			lambdas = self.sigma * ((1 / 2) * (1 - label_diff_matrix) - \
									tf.nn.sigmoid(-self.sigma * pred_diff_matrix))

			with tf.name_scope("ndcg"):
				cg_discount = tf.math.log(1.0 + index)
				rel = 2 ** labels - 1
				sorted_rel = 2 ** sorted_labels - 1
				dcg_m = rel / cg_discount
				dcg = tf.reduce_sum(dcg_m)

				stale_ij = tf.tile(dcg_m, [1, batch_size])
				new_ij = rel / tf.transpose(cg_discount)
				stale_ji = tf.transpose(stale_ij)
				new_ji = tf.transpose(new_ij)
				dcg_new = dcg - stale_ij + new_ij - stale_ji + new_ji
				dcg_max = tf.reduce_sum(sorted_rel / cg_discount)
				ndcg_delta = tf.math.abs(dcg_new - dcg) / dcg_max

			# print("ndcg :- ", ndcg_delta)
			# print("lambdas :- ", lambdas)

			lambdas = lambdas * ndcg_delta

		return lambdas

	def set_train_test_function(self, graph_mode):
		if graph_mode:
			print("Running Model in graph mode.............")
			self.test_fuc = self.test_step
			self.train_fuc = self.factorized_train_step
		else:
			print("Running Model in eager mode.............")
			self.test_fuc = self._test_step
			self.train_fuc = self._factorized_train_step
