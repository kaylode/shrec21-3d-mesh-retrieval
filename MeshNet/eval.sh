#Arguments
ROOT='/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/'

FOLD=1
TASK='Shape'
NUM_FACES=20000

WEIGHT=''
DISTANCE='cosine' #[euclidean|cosine|dotprod]

# Variables
DATA_ROOT="$ROOT/datasets/dataset${TASK}/simplified_folds/fold_${FOLD}"
NPY_PATH="$ROOT/results/${FOLD}/embed_fold_${FOLD}.npy"
GT_NPY_PATH="$ROOT/results/gt.npy"
TXT_PATH="$ROOT/results/${FOLD}/f${FOLD}-distmat.txt"
QUERY_CSV="$ROOT/datasets/dataset${TASK}/annotations/${FOLD}_val.csv"
GALLERY_CSV="$ROOT/datasets/dataset${TASK}/annotations/${FOLD}_train.csv"
REPORT_CSV="$ROOT/results/${FOLD}/f${FOLD}-report.csv"

python one_hot_gt.py -t $TASK -r $ROOT
python extract_features.py -f $FOLD -w $WEIGHT -t $TASK -r $DATA_ROOT --num_faces $NUM_FACES

cd scripts
python gen_distmat.py -q $NPY_PATH -g $GT_NPY_PATH -o $TXT_PATH -m $DISTANCE --rerank
python evaluate_distmat.py -q $QUERY_CSV -g $GALLERY_CSV -o $REPORT_CSV -d $TXT_PATH