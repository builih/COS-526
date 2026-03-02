eval "$(conda shell.bash hook)"
source "$CONDA_PREFIX/etc/profile.d/conda.sh"

ENV_NAME='gs2d'

mamba create -n $ENV_NAME python=3.9 -y
conda activate $ENV_NAME

# !!! Ensure correct environment is actually activated !!! #
if [ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ]; then
    echo "CONDA_DEFAULT_ENV is $ENV_NAME"
else
    echo "Error: CONDA_DEFAULT_ENV is not $ENV_NAME"
    exit 1
fi


mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
mamba install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
mamba env update -f environment.yml

pip install -q numpy==1.26.4
