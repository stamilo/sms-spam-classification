import os
import argparse
import datetime
import config as cfg
import inputs as input
from model import model
from solver import solver



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--epoch', default=10, type=int)
	#default=0은 input_pipeline에서 batch size를 자동 조정(각 데이터 셋 수의 10%)
	parser.add_argument('--train_batch_size', default=0, type=int)
	parser.add_argument('--valid_batch_size', default=0, type=int)
	parser.add_argument('--test_batch_size', default=0, type=int)
	
	args = parser.parse_args()
	
	if args.epoch != cfg.MAX_EPOCH:
		cfg.MAX_EPOCH = args.epoch 
	if args.train_batch_size != cfg.TRAIN_BATCH_SIZE:
		cfg.TRAIN_BATCH_SIZE = args.train_batch_size 
	if args.valid_batch_size != cfg.VALID_BATCH_SIZE:
		cfg.VALID_BATCH_SIZE = args.valid_batch_size
	if args.test_batch_size != cfg.TEST_BATCH_SIZE:
		cfg.TEST_BATCH_SIZE = args.test_batch_size 
		
								
	total_train_size,total_valid_size,total_test_size,train_batch,valid_batch,test_batch=input.input_pipeline(
																												cfg.DATASET_PATH,
																												cfg.NUM_LABELS,	
																												cfg.NUM_DIM,
																												train_split_ratio=0.6,
																												valid_split_ratio=0.2,
																												test_split_ratio=0.2)																												
	print("total train set size: %d/train batch size:%d\n"%(total_train_size,cfg.TRAIN_BATCH_SIZE))
	print("total valid set size: %d/valid batch size:%d\n"%(total_valid_size,cfg.VALID_BATCH_SIZE))
	print("total test set size: %d/test batch size:%d\n"%(total_test_size,cfg.TEST_BATCH_SIZE))
	
	print ("input pipeline ready")
	net= model(cfg.NUM_LABELS,cfg.NUM_DIM)
	print ("build_networks")
	_solver=solver(net,cfg.LOGS_PATH)
	print ("train")
	max_index = _solver.train_split(
						cfg.MAX_EPOCH,
						train_batch,
						valid_batch,
						total_train_size,
						cfg.TRAIN_BATCH_SIZE,
						total_valid_size,
						cfg.VALID_BATCH_SIZE)
				
	print ("test")
	_solver.test(
				test_batch,
				total_test_size,
				cfg.TEST_BATCH_SIZE,
				max_index
				)
				
	_solver.close()
	print ("end")
	
	
	


if __name__ == '__main__':
	main()