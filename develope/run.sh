
minifile=Miniconda3-latest-Linux-x86_64.sh
salenv=salenv

[[ -f $minifile ]] || wget -nc https://repo.anaconda.com/miniconda/$minifile
[[ -d miniconda ]] || sh Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda

export PATH=./miniconda/bin:$PATH

#which python
#conda env list

if [ $(conda env list | grep $salenv | wc -l) -ne 1 ]; then
conda env update --file env.yml --name $salenv
fi

eval "$(conda shell.bash hook)"
conda activate $salenv

#conda env list

# One needs to create the "dataset" directory and make a copy of the data into seprated sub-directorirs 

python try0.py --dataset dataset --lx 128 --ly 128 --n_sample 3 --epochs 400 --BS 32 --restart



