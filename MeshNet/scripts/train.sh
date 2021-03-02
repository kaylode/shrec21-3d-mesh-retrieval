# Arguments
ROOT='/home/pmkhoi/source/shrec21/retrieval/MeshNet/'
cd $ROOT

FOLD=1
TASK='Culture'
NUM_FACES=30000

SAVED_PATH="/home/pmkhoi/source/shrec21/retrieval/MeshNet/ckpt_root/shape/30k-fold${FOLD}"

DATA_ROOT="$ROOT/datasets/dataset${TASK}/folds/fold_${FOLD}"

python train.py -r $DATA_ROOT -t $TASK --num_faces $NUM_FACES -s $SAVED_PATH