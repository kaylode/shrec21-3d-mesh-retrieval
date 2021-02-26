ROOT='/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/'

FOLD=4
TASK='Shape'
NUM_FACES=15000

cd data
python make_dataset.py -t $TASK -f $FOLD -r $ROOT
python preprocess.py -t $TASK -f $FOLD --num_faces $NUM_FACES -r $ROOT