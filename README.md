# SimVec: Predicting polypharmacy side effects for new drugs

SimVec is a knowledge graph based model to predict polyphamracy side effects for new drugs by enhancing the knowledge graph structure with a chemistry-aware node initialization and weighted drug similarity edges.

## Data
All required data files can be downloaded from: 
https://drive.google.com/drive/folders/1_6khZG4tUs1PnEh9EJLBLxD-uOQq-tyf?usp=sharing

The following data files are available to easily reproduce paper's results: 
- Train/val/test split: polyphar_train_new_2.csv, polyphar_val_new_2.csv, polyphar_test_new_2.csv
- Enumeration of drugs and side effects to be used in code: ent_maps.csv, rel_maps.csv
- Single side effects: bio-decagon-mono.csv
- Precomputed molecular descriptors: mol_decsriptors_191.csv
- Precomputed morgan fingerprints: chemical_embed_morgan_fp_3_100.csv
- Precomputed nearest neighbours for drugs: weak_closest.pickle

## Usage

### Paper's results

To train and test SimVec_full model use the following command:

```
python run_simvec.py +experiment=simvec_full
```

### Other experiments

* To run a specific experiment (SimVec version) you need to pass an experiment name
* All available experiments can be found in the folder config/experiment. You need to pass a filename (without `.yaml`) to cmd:
```
python run_simvec.py +experiment={experiment_name}
```
Example:
```
python run_simvec.py +experiment=simvec_se
```

### Changing specific parameters of the experiments
* One can change a specific argument in the corresponding config file (just edit it) or pass a value to cmd
```
python run_simvec.py +experiment=simvec_se {arg1_name=value1} {arg2_name=value2}
```
Example:
```
python run_simvec.py +experiment=simvec_se params.epoch=50 run_args.gpu=False
```
