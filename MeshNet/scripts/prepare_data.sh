# Arguments
ROOT='/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/'
cd $ROOT

FOLD=4
TASK='Shape'
NUM_FACES=20000

DATA_ROOT="$ROOT/datasets/dataset${TASK}/objects/train"

cd data
python preprocess.py -f $DATA_ROOT --num_faces $NUM_FACES
python make_dataset.py -t $TASK -f $FOLD -r $ROOT