# Agruments
ROOT='/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/'
cd $ROOT

FOLD=4
TASK='Shape'
NUM_FACES=20000

WEIGHT='/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/ckpt_root/shape/fold4-best/MeshNet_fold4_best_0.8333333333333334.pkl'
DISTANCE='euclidean' #[euclidean|cosine|dotprod]

# Variables
TEST_ROOT="$ROOT/datasets/dataset${TASK}/objects/test"
TEST_SIMPLIFIED_ROOT="$ROOT/datasets/dataset${TASK}/objects/test_simplified"
NPY_PATH="$ROOT/results/test/embed_${FOLD}.npy"
GT_NPY_PATH="$ROOT/results/gt.npy"
TXT_PATH="$ROOT/results/test/test-distmat_f${FOLD}.txt"

cd data
python preprocess.py -f $TEST_ROOT --num_faces $NUM_FACES

cd ..
python test.py -t $TASK -w $WEIGHT -r $TEST_SIMPLIFIED_ROOT --num_faces $NUM_FACES -f $FOLD

cd utils
python one_hot_gt.py -t $TASK -r $ROOT -o $NPY_PATH

python gen_distmat.py -q $NPY_PATH -g $GT_NPY_PATH -o $TXT_PATH -m $DISTANCE --rerank
