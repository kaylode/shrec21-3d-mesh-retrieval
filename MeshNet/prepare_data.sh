FOLD=3
TASK='Shape'
NUM_FACES=30000

cd data
python make_dataset.py -t $TASK -f $FOLD 
python preprocess.py -t $TASK -f $FOLD --num_faces $NUM_FACES