ROOT='/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/'

FOLD=-1
TASK='Shape'
NUM_FACES=15000

WEIGHT='/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/ckpt_root/shape/fold4-best/MeshNet_fold4_best_0.8333333333333334.pkl'
NPY_PATH="$ROOT/results/test/embed.npy"
GT_NPY_PATH="$ROOT/results/gt.npy"
TXT_PATH="$ROOT/results/test/test-distmat.txt"

cd data
python preprocess.py -r $ROOT -t $TASK -f $FOLD --num_faces $NUM_FACES

cd ..
python one_hot_gt.py -t $TASK -r $ROOT
python test.py -t $TASK -w $WEIGHT

cd scripts
python gen_distmat.py -q $NPY_PATH -g $GT_NPY_PATH -o $TXT_PATH