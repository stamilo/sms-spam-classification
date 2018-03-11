import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import pandas as pd
import numpy as np
import config as cfg

	
#(라벨,콘텐츠(etc 음성,텍스트)) 데이터 읽기	=> trainset,validset, test으로 나뉘어짐 / ratio는 다 합쳐서 1이 되어야 됌 
def read_label_file(file='./spam.csv',train_split_ratio=0.6,valid_split_ratio=0.2,test_split_ratio=0.2):
	df = pd.read_csv(file, encoding='latin-1')
	df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
	df = df.rename(columns = {'v1':'Class','v2':'Text'})
	df = df.replace(['ham','spam'],[0, 1])
	size=len(df.Class)
	
	sum_ratio=train_split_ratio+valid_split_ratio+test_split_ratio
	
	assert  abs(1.0000 - sum_ratio) <= 0.0001, 'ratio의 합(%f)이 1이 아닙니다.'%(sum_ratio) 

	
	train_size=int(size*train_split_ratio)
	train_split_range=train_size
	train_contents=df.Text.tolist()[0:train_split_range]
	train_label=df.Class.tolist()[0:train_split_range]
	
	valid_size=int(size*valid_split_ratio)
	valid_split_range=valid_size+train_split_range
	valid_contents=df.Text.tolist()[train_split_range:valid_split_range]
	valid_label=df.Class.tolist()[train_split_range:valid_split_range]
	
	test_size=int(size*test_split_ratio)
	test_split_range=test_size+valid_split_range
	test_contents=df.Text.tolist()[valid_split_range:test_split_range]
	test_label=df.Class.tolist()[valid_split_range:test_split_range]
	
	
	return train_size,valid_size,test_size,train_contents,train_label,valid_contents,valid_label,test_contents,test_label
	
	
	
#list의 원소의 길이 중 최대 길이 추출 	
def extract_max_dim_and_tokenizer(data):
	tok = tf.keras.preprocessing.text.Tokenizer()	
	tok.fit_on_texts(data)
	max_dim=len(tok.word_index)+1
	return max_dim,tok
	
	
def preprocess(data,label,num_class,tok):

	#data를 sequences로 변환
	contents = tok.texts_to_matrix(data)
	# string을 tensor로 변환, type=float32
	label = tf.one_hot(label,depth=num_class,on_value=1,off_value=0,axis=-1)
	return contents,label
	
def create_queue(contents,labels):

	#입력 큐 생성, 데이터 섞기(shuffle)=true 
	input_queue = tf.train.slice_input_producer(
												[contents, labels],
												shuffle=True)
	#입력 큐 대입(데이터,라벨)
	content = input_queue[0]
	label = input_queue[1]
	
	return content,label						
						
# #입력 파이프라인 함수 
def input_pipeline(dataset_path,num_class,num_dim,train_split_ratio=0.6,valid_split_ratio=0.2,test_split_ratio=0.2):

	#read_label_file_split(dataset_path)
	total_train_size,total_valid_size,total_test_size,train_contents,train_labels,valid_contents,valid_labels,test_contents,test_labels = read_label_file(dataset_path,train_split_ratio,valid_split_ratio,test_split_ratio)
	
	dataset=train_contents+valid_contents+test_contents
	
	#max_dim, Tokenizer 추출
	#(1)한개를 분류할때는 Tokenization을 위해서 단어에 대한 사전(dictionary) 파일이 필요함 
	#(2)사전(dictionary)파일이 없을 경우 학습 데이터와 임의로 설정한 num_words를 설정(현재 코드에선 고려하지 않고 있음)
	max_dim,tok=extract_max_dim_and_tokenizer(dataset)
	cfg.NUM_DIM=max_dim
	
	#batch size  = 전체 데이터셋 수의 10%으로 설정 
	if(cfg.TRAIN_BATCH_SIZE==0):
		cfg.TRAIN_BATCH_SIZE=(int)(total_train_size*0.1)
	if(cfg.VALID_BATCH_SIZE==0):
		cfg.VALID_BATCH_SIZE=(int)(total_valid_size*0.1) 
	if(cfg.TEST_BATCH_SIZE==0):
		cfg.TEST_BATCH_SIZE=(int)(total_test_size*0.1) 
		
	#tokenization, padding 추가 전처리 수행 => train,valid,test set 각각 전처리
	train_contents,train_labels=preprocess(train_contents,train_labels,num_class,tok)
	valid_contents,valid_labels=preprocess(valid_contents,valid_labels,num_class,tok)
	test_contents,test_labels=preprocess(test_contents,test_labels,num_class,tok)
	
	
	
	#큐 생성  => train,valid,test  => train,valid,test set 각각 큐 생성 
	train_content, train_label=create_queue(train_contents,train_labels)
	valid_content, valid_label=create_queue(valid_contents,valid_labels)
	test_content, test_label=create_queue(test_contents,test_labels)
	
	

	
	# train,valid,test batch 객체 생성
	train_batch = tf.train.batch(
							[train_content, train_label],
							batch_size=cfg.TRAIN_BATCH_SIZE
						)
						
	valid_batch = tf.train.batch(
							[valid_content, valid_label],
							batch_size=cfg.VALID_BATCH_SIZE
						)		
						
	test_batch = tf.train.batch(
							[test_content, test_label],
							batch_size=cfg.TEST_BATCH_SIZE
						)			
					
	
	#  데이터셋 수, batch 반환
	return total_train_size,total_valid_size,total_test_size,train_batch,valid_batch,test_batch
						
														
														
														