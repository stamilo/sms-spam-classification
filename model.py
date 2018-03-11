import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope

#batch norm
def Batch_Norm(x, training, scope="bn"):
	with arg_scope([batch_norm],
					scope=scope,
					updates_collections=None,
					decay=0.9,
					center=True,
					scale=True,
					zero_debias_moving_mean=True):
		return tf.cond(training,
						lambda : batch_norm(inputs=x, is_training=training, reuse=None),
						lambda : batch_norm(inputs=x, is_training=training, reuse=True))
	
	

#모델 
class model(object):	
		
	def __init__(self,num_class,input_dim):
		self.num_class = num_class
		self.input_dim = input_dim
		self.training=tf.placeholder(tf.bool)
		self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name = 'x') 
		self.y_target = tf.placeholder(tf.float32, shape=[None, self.num_class], name = 'y_target')
		self.logits = self.build_network(x=self.x,input_dim=self.input_dim,num_class=self.num_class)

		
	def RB_layer(self,x, input_dim,output_size,scope):
		xavier_initializer = tf.contrib.layers.xavier_initializer()
		W = tf.Variable(xavier_initializer([input_dim, output_size]))
		b = tf.Variable(xavier_initializer([output_size]))
		_x = tf.nn.relu(tf.matmul(x, W) + b)
		_x = Batch_Norm(_x,self.training, scope="bn_"+scope)
		return _x
		
	def build_network(self,x,input_dim,num_class):
	
		xavier_initializer = tf.contrib.layers.xavier_initializer()
		
		# reshape input data
		_x = tf.reshape(x,[-1,input_dim],name="x_data")
		shape = _x.get_shape().as_list()
		input_num= shape[1]
				
		# Build a fully connected layer
		_x=self.RB_layer(_x,input_num,512,scope="layer1")
		shape = _x.get_shape().as_list()
		input_num= shape[1]
		
		
		# Build a fully connected layer
		_x=self.RB_layer(_x,input_num,256,scope="layer2")
		shape = _x.get_shape().as_list()
		input_num= shape[1]

		# Build a fully connected layer
		W = tf.Variable(xavier_initializer([input_num, num_class]))
		b = tf.Variable(xavier_initializer([num_class]))

		# Build a output layer
		return tf.matmul(_x, W) + b		
		
		
		
