# Arguments
ROOT='/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/'
cd $ROOT

FOLD=4
TASK='Shape'
NUM_FACES=20000

SAVED_PATH="/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/ckpt_root/shape/20k-fold${FOLD}"

DATA_ROOT="$ROOT/datasets/dataset${TASK}/folds/fold_${FOLD}"

python train.py -r $DATA_ROOT -t $TASK --num_faces $NUM_FACES -s $SAVED_PATH