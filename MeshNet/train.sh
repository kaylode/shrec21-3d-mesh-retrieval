# Arguments
ROOT='/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/'

FOLD=1
TASK='Shape'
NUM_FACES=20000

SAVED_PATH="/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/ckpt_root/shape/20k-fold1"

DATA_ROOT="$ROOT/datasets/dataset${TASK}/simplified_folds/fold_${FOLD}"

python train.py -r $DATA_ROOT -t $TASK --num_faces $NUM_FACES -s $SAVED_PATH -w $WEIGHT