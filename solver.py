
import tensorflow as tf
import config as cfg
#solver 클래스 : 모델의 학습, 테스트를 관리 
class solver(object):

	def __init__(self,net,logs_path,learning_rate=0.1):
		self.logs_path=logs_path
		self.net = net
		#cost, train_step, correct_prediction, accuracy, training_flag
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.net.logits, labels=self.net.y_target))
		self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
		self.correct_prediction = tf.equal(tf.argmax(self.net.logits, 1), tf.argmax(self.net.y_target, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float")) 
		#웨이트 저장 
		self.t_vars = tf.trainable_variables()
		self.saver = tf.train.Saver(max_to_keep=None,var_list=self.t_vars)
	
		
		#로그 저장 
		self.f=open(cfg.LOGS_PATH+'logs.txt', 'a')
		
		#GPU 설정
		self.config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True))
		self.sess = tf.Session(config=self.config)
	
		#sigle op에 모든 summery merg함 
		self.merged_op = tf.summary.merge_all()
		self.writer=tf.summary.FileWriter(logs_path, self.sess.graph)
		
		# 쓰레드 큐 초기화
		self.coord = tf.train.Coordinator()
		self.threads = tf.train.start_queue_runners(sess=self.sess,coord=self.coord)

		
		
	#loss와 auccuracy 연산
	def cal_loss_accuracy(self,epoch,total_size,batch_size,input_batch,loss_tag,accuracy_tag,training):
		cal_loss=0
		cal_acc=0
		
		assert (total_size > batch_size), '배치 사이즈(%d)가 데이터 셋 크기(%d)보다 작습니다'%(batch_size,total_size)
		
		total_iteration = float(total_size/batch_size)
		
		
		if total_iteration > int(total_size/batch_size) :
			total_iteration=int(total_iteration) + 1
		else:
			total_iteration=int(total_iteration)
				
		for it in range(total_iteration):
			_input_batch=self.sess.run([tf.cast(input_batch[0], tf.float32),tf.cast(input_batch[1], tf.float32)])
			feed_dicts = {
				self.net.x: _input_batch[0],
				self.net.y_target : _input_batch[1],
				self.net.training: training
			}
			loss, acc = self.sess.run(
				[self.cost,self.accuracy],
				feed_dict=feed_dicts)
			cal_loss += loss  
			cal_acc += acc
			
		cal_loss/=float(total_iteration)
		cal_acc/=float(total_iteration)
		
		_summary = tf.Summary(value=[tf.Summary.Value(tag=loss_tag, simple_value=cal_loss),
								  tf.Summary.Value(tag=accuracy_tag, simple_value=cal_acc)])

		self.writer.add_summary(_summary,epoch)
		return cal_loss,cal_acc
				
		

	#train,vaild 완벽히 불리할 train 함수 => 현재 사용 x
	def train_split(self,max_epoch,train_batch,valid_batch,total_train_size,train_batch_size,total_valid_size,valid_batch_size):
		print("Session started!")
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)

		display_step=1
		max_accuracy=0
		max_index=0
		
		for epoch in range(0,max_epoch+1):
			avg_cost = 0
			assert (total_train_size > train_batch_size) , '배치 사이즈(%d)가 데이터 셋 크기(%d)보다 작습니다.'%(train_batch_size,total_train_size)
						
			total_batch = float(total_train_size/train_batch_size)
			
			if total_batch > int(total_train_size/train_batch_size) :
				total_batch=int(total_batch) + 1
			else:
				total_batch=int(total_batch)
			
			for step in range(total_batch):
				_train_batch=self.sess.run([tf.cast(train_batch[0], tf.float32),tf.cast(train_batch[1], tf.float32)])
				if epoch!=0:
					_,c=self.sess.run([self.train_step,self.cost] , feed_dict={self.net.x: _train_batch[0], self.net.y_target: _train_batch[1], self.net.training:True})
					avg_cost += c / total_batch
				else:
					c=self.sess.run(self.cost ,feed_dict={self.net.x: _train_batch[0], self.net.y_target: _train_batch[1], self.net.training:False})
					avg_cost += c / total_batch

			# Display logs per epoch step
			if epoch % display_step == 0:
				print ("Epoch:", '%04d' % (epoch), "cost=","{:.9f}".format(avg_cost))
				train_loss,train_acc=self.cal_loss_accuracy(epoch,total_train_size,train_batch_size,train_batch,"training_loss","training_accuracy",False)
				valid_loss,valid_acc=self.cal_loss_accuracy(epoch,total_valid_size,valid_batch_size,valid_batch,"validation_loss","validation_accuracy",False)	
				#print data / save data
				line = "epoch: %d/%d, train_loss: %.9f, train_acc: %.9f, valid_loss: %.9f, valid_acc: %.9f \n" % (
					epoch, max_epoch, train_loss, train_acc, valid_loss, valid_acc)
				print(line)
				self.f.write(line)
				if valid_acc>=max_accuracy:
					max_accuracy=valid_acc
					max_index=epoch				
				self.saver.save(self.sess,"./weight/w",epoch)
				
		line = "max_acc: %.9f, max_index:%d\n"%(max_accuracy,max_index)
		print(line)
		self.f.write(line)
		return max_index
		
	#train함수 => 후에 완벽히 기능 분리할 예정임 	
	def train(self,max_epoch,train_batch,total_train_size,train_batch_size):
		print("Session started!")
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)


		display_step=1
		max_accuracy=0
		max_index=0
		
		for epoch in range(0,max_epoch+1):
			avg_cost = 0
			total_batch=(int)(total_train_size/train_batch_size)
			for step in range(total_batch):
				train_batch=self.sess.run([tf.cast(train_batch[0], tf.float32),tf.cast(train_batch[1], tf.float32)])
				if epoch!=0:
					_,c=self.sess.run([self.train_step,self.cost] , feed_dict={self.net.x: train_batch[0], self.net.y_target: train_batch[1], self.net.training:True})
					# Compute average loss
					avg_cost += c / total_batch
				else:
					c=self.sess.run(self.cost ,feed_dict={self.net.x: train_batch[0], self.net.y_target: train_batch[1], self.net.training:False})
					avg_cost += c / total_batch

			# Display logs per epoch step
			if epoch % display_step == 0:
				print ("Epoch:", '%04d' % (epoch), "cost=","{:.9f}".format(avg_cost))
				train_loss,train_acc=self.cal_loss_accuracy(epoch,total_train_size,train_batch_size,train_batch,"training_loss","training_accuracy",False)	
				#print data / save data
				line = "epoch: %d/%d, train_loss: %.9f, train_acc: %.9f\n" % (
					epoch, max_epoch, train_loss, train_acc)
				print(line)
				self.f.write(line)
				if train_acc>=max_accuracy:
					max_accuracy=train_acc
					max_index=epoch				
				self.saver.save(self.sess,"./weight/w",epoch)
				
		line = "max_acc: %.9f, max_index:%d\n"%(max_accuracy,max_index)
		print(line)
		self.f.write(line)
		return max_index	
	#test함수	
	def test(self,test_batch,total_test_size,test_batch_size,max_index):
		self.saver.restore(self.sess,"./weight/w-%d"%(max_index))
		test_loss,test_acc=self.cal_loss_accuracy(1,total_test_size,test_batch_size,test_batch,"test_loss","test_accuracy",False)
		line = "test_loss: %.9f, test_acc:%.9f, max_index:%d\n"%(test_loss,test_acc,max_index)
		print(line)
		self.f.write(line)
		
	#모든 객체 종료, 메모리 해제 
	def close(self):
		self.f.close()
		self.writer.close()
		self.coord.request_stop()
		self.coord.join(self.threads)
		self.sess.close()
		print("close")
		
		