# drugs-effects

## Data: 
https://drive.google.com/drive/folders/1-UkDGitQ_MzwIuz19cdlxRsrSPsawjDY?usp=sharing

## Usage
```
python run_trivec.py +experiment={experiment_name}
```
* To run scepific experiment (ChemVec version) you need to pass experiment name
* All available experiments can be found in folder config/experiment. You need to pass filename (without `.yaml`) to cmd

Example:
```
python run_trivec.py +experiment=chemvec_se
```

* One can change specific argument in config file (just edit it) or pass value of specific config file argumet to cmd
```
python run_trivec.py +experiment=chemvec_se {arg1_name=value1} {arg2_name=value2}
```

Example:
```
python run_trivec.py +experiment=chemvec_se params.epoch=50 run_args.gpu=False
```
