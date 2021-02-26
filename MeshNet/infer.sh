# Agruments
ROOT='/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/'

FOLD=-1
TASK='Shape'
NUM_FACES=15000
WEIGHT='/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/ckpt_root/shape/test/MeshNet_best_0.8257575757575758.pkl'

# Variables
TEST_ROOT="$ROOT/datasets/dataset${TASK}/simplified_test"
NPY_PATH="$ROOT/results/test/embed.npy"
GT_NPY_PATH="$ROOT/results/gt.npy"
TXT_PATH="$ROOT/results/test/test-distmat.txt"

cd data
python preprocess.py -r $ROOT -t $TASK -f $FOLD --num_faces $NUM_FACES

cd ..
python one_hot_gt.py -t $TASK -r $ROOT
python test.py -t $TASK -w $WEIGHT -r $TEST_ROOT --num_faces $NUM_FACES

cd scripts
python gen_distmat.py -q $NPY_PATH -g $GT_NPY_PATH -o $TXT_PATH