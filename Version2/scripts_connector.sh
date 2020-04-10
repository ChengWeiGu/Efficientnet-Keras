set -e

CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate efficientnet-david

cd ./d0.7k_pn1.9
python efficientnet_run_Ver2.py \
	--train \
	--model_size="B0" \
	--imag_size=224 \
	--num_cls=2

python efficientnet_run_Ver2.py \
	--weights="efficientnetB0_last.h5" \
	--model_size="B0" \
	--imag_size=224 \
	--num_cls=2
