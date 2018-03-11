import os

#데이터셋 경로 위치
DATASET_PATH      = "./spam.csv"

#텐서보드 저장경로
LOGS_PATH	= "logs/"

#모델의 입력 차원=> 현재 input pipeline에서 자동 조정 
NUM_DIM=0
#라벨 개수
NUM_LABELS		= 2
#batch size 모두 0 => input pipeline에서 자동 조정 
TRAIN_BATCH_SIZE	= 0
VALID_BATCH_SIZE	= 0
TEST_BATCH_SIZE		= 0
#epoch 
MAX_EPOCH=10


