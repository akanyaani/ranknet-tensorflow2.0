import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from layer_norm import LayerNormalization
from metrics import ndcg_at_k

train_step_signature = [
	tf.TensorSpec(shape=(None, 136), dtype=tf.float32),
	tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
]


class BaseTFModel(tf.keras.Model):
	def __init__(self):
		super(BaseTFModel, self).__init__()
		self.optimizer = None
		self.train_writer = None
		self.test_writer = None
		self.ckpt_manager = None
		self.ndcg = 0.0
		self.train_fuc = None
		self.test_fuc = None

	def create_optimizer(self, optimizer_type):
		with tf.name_scope("optimizer"):
			if optimizer_type == "adam":
				self.optimizer = tf.keras.optimizers.Adam(self.learning_rate,
														  beta_1=0.9,
														  beta_2=0.999,
														  epsilon=1e-9)
			elif optimizer_type == "adadelta":
				self.optimizer = tf.keras.optimizers.Adadelta(self.learning_rate)
			elif optimizer_type == "rms":
				self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
			else:
				self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)
			return self.optimizer

	def create_checkpoint_manager(self, checkpoint_path, max_to_keep=5, load_model=True):
		with tf.name_scope('checkpoint_manager'):
			ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
			self.ckpt_manager = tf.train.CheckpointManager(ckpt,
														   checkpoint_path,
														   max_to_keep=max_to_keep)

			if load_model:  # If want to load trained weights
				ckpt.restore(self.ckpt_manager.latest_checkpoint)
				print('Latest checkpoint restored..............')
			else:
				print("Initializing model from scratch.........")

	def create_summary_writer(self, summary_path):
		train_summary_path = summary_path + "/train"
		test_summary_path = summary_path + "/test"

		with tf.name_scope('summary'):
			self.train_writer = tf.summary.create_file_writer(train_summary_path)
			self.test_writer = tf.summary.create_file_writer(test_summary_path)

			return self.train_writer, self.test_writer

	@staticmethod
	def _get_ndcg(target, pred_score):

		target = tf.reshape(target, [-1])
		zpd = list(zip(target.numpy(), pred_score.numpy()))
		zpd.sort(key=lambda x: x[1], reverse=True)
		pred_rank, _ = list(zip(*zpd))

		test_ndcg_5 = ndcg_at_k(list(pred_rank), 5)
		test_ndcg_20 = ndcg_at_k(list(pred_rank), 20)

		return test_ndcg_5, test_ndcg_20

	@staticmethod
	def _log_scalar_summary(writer, step, scalar_name, scalar_value, log_freq=100):
		if step % log_freq == 0:
			with writer.as_default():
				tf.summary.scalar(scalar_name, scalar_value, step=step)

	@staticmethod
	def _log_model_summary_data(writer, step, loss, ndcg5, ndcg20):
		with writer.as_default():
			with tf.name_scope('RankMetrics'):
				tf.summary.scalar("loss", loss, step=step)
				tf.summary.scalar("ndcg@5", ndcg5, step=step)
				tf.summary.scalar("ndcg@20", ndcg20, step=step)

	@staticmethod
	def _log_model_data(log_type, step, loss, accuracy=0.0):
		if step % 100 == 0:
			print('Step {} {}_Loss {:.4f}, Accuracy {:.4f}'.format(
				step, log_type, loss, accuracy))


class LTRModelRanknet(BaseTFModel):
	def __init__(self,
				 activation=tf.nn.relu,
				 learning_rate=1e-3,
				 sigma=1.0,
				 dr_rate=0.25,
				 grad_clip=True,
				 clip_value=1.0,
				 ranknet_type='norm'):
		super(LTRModelRanknet, self).__init__()

		self.learning_rate = learning_rate
		self.sigma = sigma
		self.activation = activation
		self.dr_rate = dr_rate
		self.grad_clip = grad_clip
		self.clip_value = clip_value
		self.ranknet_type = ranknet_type

		self.ln1 = LayerNormalization()

		self.dense1 = Dense(256,
							activation=self.activation)

		self.ln2 = LayerNormalization()
		self.dense2 = Dense(128,
							activation=self.activation)

		self.fc_drop = tf.keras.layers.Dropout(self.dr_rate)
		self.ln3 = LayerNormalization()
		self.output_layer = Dense(1, activation=tf.identity, use_bias=False)

	def call(self, inputs, training=False):
		# import pdb;
		# pdb.set_trace()
		inputs = tf.cast(inputs, tf.float32)

		output = self.dense1(self.ln1(inputs))
		output = self.dense2(self.ln2(output))

		output = self.fc_drop(self.ln3(output), training=training)
		score = self.output_layer(output)
		return score

	def _matching_function(self, x1, x2, training):
		score1 = self(x1, training)
		score2 = self(x2, training)

		return score1, score2

	@staticmethod
	def _get_lambda_scaled_derivative(tape, score, Wk, lambdas):
		"""https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
		https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
		∂si/∂wk−∂sj/∂wk In this method calculating this as explained in paper."""

		# import pdb;
		# pdb.set_trace()
		# print("\nShape of Wk :- ", tf.shape(Wk))
		# print("\nShape of score :- ", tf.shape(score))

		# print("\nShape of lambdas :- ", tf.shape(lambdas))
		dsi_dWk = tape.jacobian(score, Wk, experimental_use_pfor=False)  # ∂si/∂wk
		dsi_dWk_minus_dsj_dWk = tf.expand_dims(dsi_dWk, 1) - tf.expand_dims(dsi_dWk, 0)  # ∂si/∂wk−∂sj/∂wk
		# print("\nShape of dsi_dWk :- ", tf.shape(dsi_dWk))
		# print("\nShape of dsi_dWk_minus_dsj_dWk :- ", tf.shape(dsi_dWk_minus_dsj_dWk))
		shape = tf.concat([tf.shape(lambdas),
						   tf.ones([tf.rank(dsi_dWk_minus_dsj_dWk) - tf.rank(lambdas)],
								   dtype=tf.int32)], axis=0)

		# (1/2(1−Sij)−1/1+eσ(si−sj))(∂si/∂wk−∂sj/∂wk)
		# print("\nShape :- ", shape)
		grad = tf.reshape(lambdas, shape) * dsi_dWk_minus_dsj_dWk
		grad = tf.reduce_mean(grad, axis=[0, 1])
		# print("\nShape of grad :- ", tf.shape(grad))
		return grad

	@staticmethod
	def _get_ranknet_loss(pred_score, real_score, name='ranknet_loss'):
		with tf.name_scope(name):
			diff_matrix = real_score - tf.transpose(real_score)

			label = tf.maximum(tf.minimum(1., diff_matrix), -1.)
			real_label = (1 + label) / 2

			pred_diff_matrix = pred_score - tf.transpose(pred_score)
			pred_label = tf.nn.sigmoid(pred_diff_matrix)

			loss = -real_label * tf.math.log(pred_label) - \
				   (1 - real_label) * tf.math.log(1 - pred_label)

		return loss

	def _get_lambdas(self, pred_score, labels):
		"""https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
		As explained in equation 3
		(1/2(1−Sij)−1/1+eσ(si−sj))"""

		diff_matrix = labels - tf.transpose(labels)
		label_diff_matrix = tf.maximum(tf.minimum(1., diff_matrix), -1.)
		# print(label_diff_matrix)
		pred_diff_matrix = pred_score - tf.transpose(pred_score)
		# print(pred_diff_matrix)
		lambdas = self.sigma * ((1 / 2) * (1 - label_diff_matrix) - \
								tf.nn.sigmoid(-self.sigma * pred_diff_matrix))

		return lambdas

	def _train_step(self, inputs, target):
		# import pdb;
		# pdb.set_trace()
		with tf.GradientTape() as tape:
			pred_score = self(inputs, training=tf.constant(True))
			loss = tf.reduce_mean(self._get_ranknet_loss(pred_score, target))

		with tf.name_scope("gradients"):
			gradients = tape.gradient(loss, self.trainable_variables)
			if self.grad_clip:
				gradients = [(tf.clip_by_value(grad, -self.clip_value, self.clip_value))
							 for grad in gradients]
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		step = self.optimizer.iterations
		return step, loss, tf.squeeze(pred_score)

	def _factorized_train_step(self, inputs, target):
		# import pdb ; pdb.set_trace()
		with tf.GradientTape(persistent=True) as tape:
			tape.watch(inputs)
			pred_score = self(inputs, training=tf.constant(True))
			loss = tf.reduce_mean(self._get_ranknet_loss(pred_score, target))
			lambdas = self._get_lambdas(pred_score, target)
			pred_score = tf.reshape(pred_score, [-1])

		gradients = [self._get_lambda_scaled_derivative(tape, pred_score, Wk, lambdas) \
					 for Wk in self.trainable_variables]

		if self.grad_clip:
			gradients = [(tf.clip_by_value(grad, -self.clip_value, self.clip_value))
						 for grad in gradients]

		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
		step = self.optimizer.iterations

		return step, loss, pred_score

	@tf.function(input_signature=train_step_signature)
	def train_step(self, inputs, target):
		return self._train_step(inputs, target)

	@tf.function(input_signature=train_step_signature)
	def factorized_train_step(self, inputs, target):
		return self._factorized_train_step(inputs, target)

	def _test_step(self, inputs, target):
		pred_score = self(inputs, training=tf.constant(False))
		loss = tf.reduce_mean(self._get_ranknet_loss(pred_score, target))
		return loss, tf.squeeze(pred_score)

	@tf.function(input_signature=train_step_signature)
	def test_step(self, inputs, target):
		return self._test_step(inputs, target)

	@staticmethod
	def load_model(model, model_path):
		ckpt = tf.train.Checkpoint(model=model)
		ckpt_manager = tf.train.CheckpointManager(ckpt, model_path, max_to_keep=1)
		ckpt.restore(ckpt_manager.latest_checkpoint)
		print("Gpt2 Weights loaded..........................")

	def _save_model(self, ndcg):
		if ndcg > self.ndcg:
			ckpt_save_path = self.ckpt_manager.save()
			self.ndcg = ndcg
			print('Saving checkpoint at {}'.format(ckpt_save_path))

	def _init_comp_graph(self, step=0, name="gpt2_LTR"):
		with self.train_writer.as_default():
			tf.summary.trace_export(
				name=name,
				step=step,
				profiler_outdir=self.log_dir)

	def fit(self, dataset, graph_mode=False):
		if graph_mode:
			print("Running Model in graph mode......")
			self.test_fuc = self.test_step
			if self.ranknet_type == "norm":
				self.train_fuc = self.train_step
			else:
				self.train_fuc = self.factorized_train_step
		else:
			print("Running Model in eager mode......")
			self.test_fuc = self._test_step
			if self.ranknet_type == "norm":
				self.train_fuc = self._train_step
			else:
				self.train_fuc = self._factorized_train_step

		assert len(dataset) == 2
		train_dataset, test_dataset = dataset
		if graph_mode:
			tf.summary.trace_on(graph=True, profiler=True)
		for (count, (q_id, inputs, target)) in enumerate(train_dataset):
			step, train_loss, score = self.train_fuc(inputs, target)

			if step % 1 == 0:
				ndcg5, ndcg20 = self._get_ndcg(target, score)
				self._log_model_summary_data(self.train_writer,
											 step,
											 train_loss,
											 ndcg5,
											 ndcg20)

			if step == 1:
				if graph_mode:
					self._init_comp_graph()

			if step % 1000 == 0:
				losses = []
				ndcg_5 = []
				ndcg_20 = []

				for (test_step, (qid, inputs_test, target_test)) in enumerate(test_dataset):
					test_loss, pred_score = self.test_fuc(inputs_test, target_test)

					ndcg5, ndcg20 = self._get_ndcg(target_test, pred_score)
					losses.append(test_loss)
					ndcg_5.append(ndcg5)
					ndcg_20.append(ndcg20)

					if test_step == 100:
						break

				test_loss = np.mean(np.array(losses))
				test_ndcg_5 = np.mean(np.array(ndcg_5))
				test_ndcg_20 = np.mean(np.array(ndcg_20))

				self._log_model_summary_data(self.test_writer,
											 step,
											 test_loss,
											 test_ndcg_5,
											 test_ndcg_20)

				self._save_model(test_ndcg_20)
