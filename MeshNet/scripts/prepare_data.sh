# Arguments
ROOT='/home/pmkhoi/source/shrec21/retrieval/MeshNet/'
cd $ROOT

FOLD=1
TASK='Culture'
NUM_FACES=30000

DATA_ROOT="$ROOT/datasets/dataset${TASK}/objects/train"

cd data
python preprocess.py -f $DATA_ROOT --num_faces $NUM_FACES
python make_dataset.py -t $TASK -f $FOLD -r $ROOT