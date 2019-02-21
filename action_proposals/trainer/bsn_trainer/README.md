# Boundary Sensitive Network(BSN)
We use the same data format as the original tf repo. 
So that every module can compare or replace with it.

The whole process will cost about 20 minutes. And the AUC is 66.27, which is 
consist with the performance reported in the original paper.

Execute this command to correctly setting the environment.
```bash
export PYTHONPATH=[PROJECT_PATH]:${PYTHONPATH}
```

All the configs are in cfgs. Change essential config to run.
## Tem Train

```bash
python tem_trainer.py
```

Train the TEM model. This process will cost 10 minutes.
## Tem Test

```bash
python tem_trainer.py --runtype test
```

This file will generate tem results on training set and validation set,
It will cost about 90s.

## Generate Proposal

```bash
python pgm_proposal_generation.py
```

This file will run in multiprocess. The whole process will cost about
100s. 

## Generate PGM Features

```bash
python pgm_feature_generation.py
```

This file will run in multiprocess. The whole process will cost about
50s. 

## Train PEM

 ```bash
 python pem_train.py
 ```
 
# Post processing

This process will do soft nms to the proposals. This process will cost
about 50s.

```bash
python post_processing.py
```
This file will run in multiprocess. The whole process will cost about
180s. 

# Eval

```bash
python ../../evaluate/anet_eval_proposals.py
```