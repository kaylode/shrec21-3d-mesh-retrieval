# Agruments
ROOT='/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/'

FOLD=-1
TASK='Shape'
NUM_FACES=20000

WEIGHT=''
DISTANCE='cosine' #[euclidean|cosine|dotprod]

# Variables
TEST_ROOT="$ROOT/datasets/dataset${TASK}/objects/test"
TEST_SIMPLIFIED_ROOT="$ROOT/datasets/dataset${TASK}/objects/test_simplified"
NPY_PATH="$ROOT/results/test/embed.npy"
GT_NPY_PATH="$ROOT/results/gt.npy"
TXT_PATH="$ROOT/results/test/test-distmat.txt"

cd data
python preprocess.py -f $DATA_ROOT --num_faces $NUM_FACES

cd ..
python one_hot_gt.py -t $TASK -r $ROOT
python test.py -t $TASK -w $WEIGHT -r $TEST_SIMPLIFIED_ROOT --num_faces $NUM_FACES

cd scripts
python gen_distmat.py -q $NPY_PATH -g $GT_NPY_PATH -o $TXT_PATH -m $DISTANCE --rerank
