# Boundary Sensitive Network(BSN)

Execute this command to correctly setting the environment.
```bash
export PYTHONPATH=[PROJECT PATH]:${PYTHONPATH}
```

All the configs are in cfgs. Change essential config to run.
## Tem Train

```bash
python tem_trainer.py
```

## Tem Test

```bash
python tem_test.py
```

## Generate Proposal

```bash
python pgm_proposal_generation.py
```

This file will run in multiprocess. The whole process will cost about
50s. 

## Generate PGM Features

```bash
python pgm_feature_generation.py
```

This file will run in multiprocess. The whole process will cost about
30s. 

## Train PEM

 ```bash
 python pem_train.py
 ```
 
# Post processing

This process will do soft nms
```bash
python post_processing.py
```
This file will run in multiprocess. The whole process will cost about
180s. 