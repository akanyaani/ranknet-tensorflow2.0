from ranknet import *


class LTRModelLambdaRank(LTRModelRanknet):
	def __init__(self,
				 activation=tf.nn.relu,
				 learning_rate=1e-3,
				 sigma=1.0,
				 dr_rate=0.25,
				 grad_clip=True,
				 clip_value=0.50):

		super(LTRModelRanknet, self).__init__()

		self.learning_rate = learning_rate
		self.sigma = sigma
		self.activation = activation
		self.dr_rate = dr_rate
		self.grad_clip = grad_clip
		self.clip_value = clip_value
