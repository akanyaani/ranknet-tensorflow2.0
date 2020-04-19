from typing import Optional, Any

from utils import gelu
import tensorflow as tf

import numpy as np

train_step_signature = [
	tf.TensorSpec(shape=(None, 1536), dtype=tf.float32),
	tf.TensorSpec(shape=(None, 768), dtype=tf.float32),
	tf.TensorSpec(shape=(None, 19), dtype=tf.float32),
	tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
]


class BaseTFModel(tf.keras.Model):
	def __init__(self):
		super(BaseTFModel, self).__init__()
		self.optimizer = None
		self.ckpt_manager = None
		self.train_writer = None
		self.test_writer = None
		self.val_loss = 1000.0

	def creat_optimizer(self, optimizer_type):
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
	def _log_scalar_summary(writer, step, scalar_name, scalar_value, log_freq=100):
		if step % log_freq == 0:
			with writer.as_default():
				tf.summary.scalar(scalar_name, scalar_value, step=step)

	@staticmethod
	def _log_model_summary_data(writer, step, loss, log_freq=100):
		if step % log_freq == 0:
			with writer.as_default():
				tf.summary.scalar("loss", loss, step=step)

	@staticmethod
	def _log_model_data(log_type, step, loss, accuracy=0.0):
		if step % 100 == 0:
			print('Step {} {}_Loss {:.4f}, Accuracy {:.4f}'.format(
				step, log_type, loss, accuracy))


class LTRModelRanknet(BaseTFModel):
	def __init__(self, qu_fc_dim=1024,
				 doc_fc_dim=1024,
				 dr_rate=0.25,
				 activation=gelu,
				 learning_rate=1e-3,
				 grad_clip=True,
				 clip_value=0.50):
		super(LTRModelRanknet, self).__init__()

		self.learning_rate = learning_rate
		self.qu_fc_dim = qu_fc_dim
		self.doc_fc_dim = doc_fc_dim
		self.activation = activation
		self.dr_rate = dr_rate
		self.grad_clip = grad_clip
		self.clip_value = clip_value

		self.ln1 = LayerNormalization(1536)
		self.query_dense = DenseLayer(self.qu_fc_dim,
									  activation=self.activation)
		self.query_drop_layer = tf.keras.layers.Dropout(0.05)

		self.ln2 = LayerNormalization(768)
		self.doc_dense = DenseLayer(self.doc_fc_dim,
									activation=self.activation)
		self.doc_drop_layer = tf.keras.layers.Dropout(0.05)

		self.ln3 = LayerNormalization(1043)
		self.dense1 = DenseLayer(768,
								 activation=self.activation)

		self.ln4 = LayerNormalization(768)
		self.dense2 = DenseLayer(512,
								 activation=self.activation)

		self.ln5 = LayerNormalization(512)
		self.dense3 = DenseLayer(256,
								 activation=self.activation)

		self.fc_drop = tf.keras.layers.Dropout(self.dr_rate)
		self.ln6 = LayerNormalization(256)
		self.output_layer = DenseLayer(1, activation=tf.identity, use_bias=False)

	def call(self, query, doc, add, training):
		query = tf.cast(query, tf.float32)
		doc = tf.cast(doc, tf.float32)
		add = tf.cast(add, tf.float32)

		query = self.query_drop_layer(self.ln1(query), training=training)
		query_vec = self.query_dense(query)

		doc = self.doc_drop_layer(self.ln2(doc), training=training)
		doc_vec = self.doc_dense(doc)

		with tf.name_scope("HadmardProduct"):
			final_inp = tf.multiply(query_vec, doc_vec)

		with tf.name_scope("AddFeatureCocat"):
			final_inp = tf.concat([final_inp, add], 1)

		output = self.dense1(self.ln3(final_inp))
		output = self.dense2(self.ln4(output))
		output = self.dense3(self.ln5(output))

		output = self.fc_drop(self.ln6(output), training=training)
		score = tf.squeeze(self.output_layer(output))
		return score

	@staticmethod
	def get_factorized_ranknet_loss(pred_score, real_score):
		with tf.name_scope("ranknet_loss"):
			diff_matrix = real_score - tf.transpose(real_score)
			label = tf.maximum(tf.minimum(1., diff_matrix), -1.)
			real_label = (1 + label) / 2

			pred_diff_matrix = pred_score - tf.transpose(pred_score)
			pred_label = tf.nn.sigmoid(pred_diff_matrix)

			loss = -real_label * tf.math.log(pred_label) - \
				   (1 - real_label) * tf.math.log(1 - pred_label)

		return loss

	def _train_step(self, query, doc, add, target):
		with tf.GradientTape() as tape:
			pred_score = self(query, doc, add, training=tf.constant(True))
			loss = tf.reduce_mean(self.get_factorized_ranknet_loss(pred_score, target))

		with tf.name_scope("gradients"):
			gradients = tape.gradient(loss, self.trainable_variables)
			if self.grad_clip:
				gradients = [(tf.clip_by_value(grad, -self.clip_value, self.clip_value))
							 for grad in gradients]
			self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		step = self.optimizer.iterations
		self._log_model_summary_data(self.train_writer,
									 step,
									 loss)
		return step, loss

	@tf.function(input_signature=train_step_signature)
	def train_step(self, query, doc, add, target):
		return self._train_step(query, doc, add, target)

	def _test_step(self, query, doc, add, target):
		pred_score: Optional[Any] = self(query, doc, add, target, training=tf.constant(False))
		loss = tf.reduce_mean(self.get_factorized_ranknet_loss(pred_score, target))
		return loss

	@tf.function(input_signature=train_step_signature)
	def test_step(self, query, doc, add, target):
		return self._test_step(query, doc, add, target)

	@staticmethod
	def load_model(model, model_path):
		ckpt = tf.train.Checkpoint(model=model)
		ckpt_manager = tf.train.CheckpointManager(ckpt, model_path, max_to_keep=1)
		ckpt.restore(ckpt_manager.latest_checkpoint)
		print("Gpt2 Weights loaded..........................")

	def _save_model(self, test_loss):
		if test_loss < self.val_loss:
			ckpt_save_path = self.ckpt_manager.save()
			self.val_loss = test_loss
			print('Saving checkpoint at {}'.format(ckpt_save_path))

	def _init_comp_graph(self, step=0, name="gpt2_LTR"):
		with self.train_writer.as_default():
			tf.summary.trace_export(
				name=name,
				step=step,
				profiler_outdir=self.log_dir)

	def fit_eagerly(self, dataset):
		assert len(dataset) == 2
		train_dataset, test_dataset = dataset
		print("Running in eager mode...........................")
		for (count, (q_id, query, doc, add, target)) in enumerate(train_dataset):

			step, train_loss = self.train_step_eagerly(query, doc, add, target)
			self._log_model_data("Train", step, train_loss)

			if count % 10000 == 0:
				gc.collect()
				losses = []
				for (test_step, (q_id_t, query_t, doc_t, add_t, target_t)) in enumerate(test_dataset):
					test_loss = self.test_step_eagerly(query_t, doc_t, add_t, target_t)
					losses.append(test_loss)

					if test_step == 100:
						break

				test_loss = np.mean(np.array(losses))
				self._log_model_summary_data(self.test_writer,
											 step,
											 test_loss,
											 log_freq=tf.constant(1.0))
				self._save_model(test_loss)

	def fit(self, dataset):
		assert len(dataset) == 2
		train_dataset, test_dataset = dataset
		tf.summary.trace_on(graph=True, profiler=True)
		for (count, (q_id, query, doc, add, target)) in enumerate(train_dataset):
			step, train_loss = self.train_step(q_id, query, doc, add, target)
			if step == 1:
				self._init_comp_graph()

			if step % 10000 == 0:
				losses = []
				for (test_step, (q_id_t, query_t, doc_t, add_t, target_t)) in enumerate(test_dataset):
					test_loss = self.test_step(query_t, doc_t, add_t, target_t)
					losses.append(test_loss)
					if test_step == 100:
						break
				test_loss = np.mean(np.array(losses))
				self._log_model_summary_data(self.test_writer,
											 step,
											 test_loss,
											 log_freq=tf.constant(1.0))

				self._save_model(test_loss)


class LearningSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, d_model, warmup_steps=2000, avg_constant=0.01):
		super(LearningSchedule, self).__init__()
		self.d_model = tf.cast(d_model, tf.float32)
		self.warmup_steps = warmup_steps
		self.avg_constant = tf.cast(avg_constant, tf.float32)

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps ** -1.5)

		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) * self.avg_constant
