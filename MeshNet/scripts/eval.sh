#Arguments
ROOT='/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/'
cd $ROOT

FOLD=4
TASK='Shape'
NUM_FACES=20000

WEIGHT='/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/ckpt_root/shape/fold4-best/MeshNet_fold4_best_0.8333333333333334.pkl'
DISTANCE='cosine' #[euclidean|cosine|dotprod]

# Variables
DATA_ROOT="$ROOT/datasets/dataset${TASK}/folds/fold_${FOLD}"
NPY_PATH="$ROOT/results/${FOLD}/embed_fold_${FOLD}.npy"
GT_NPY_PATH="$ROOT/results/gt.npy"
TXT_PATH="$ROOT/results/${FOLD}/f${FOLD}-distmat.txt"
QUERY_CSV="$ROOT/datasets/dataset${TASK}/annotations/${FOLD}_val.csv"
GALLERY_CSV="$ROOT/datasets/dataset${TASK}/annotations/${FOLD}_train.csv"
REPORT_CSV="$ROOT/results/${FOLD}/f${FOLD}-report.csv"

python extract_features.py -f $FOLD -w $WEIGHT -t $TASK -r $DATA_ROOT --num_faces $NUM_FACES

cd utils
python one_hot_gt.py -t $TASK -r $ROOT -o $GT_NPY_PATH

python gen_distmat.py -q $NPY_PATH -g $GT_NPY_PATH -o $TXT_PATH -m $DISTANCE --rerank
python evaluate_distmat.py -q $QUERY_CSV -g $GALLERY_CSV -o $REPORT_CSV -d $TXT_PATH