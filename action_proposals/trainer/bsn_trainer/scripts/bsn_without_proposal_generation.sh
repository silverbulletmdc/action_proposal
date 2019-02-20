#!/usr/bin/env zsh

root_path="`pwd`/../../.."
bsn_cfg_file=$1
#which python
source activate action_proposal
export PYTHONPATH=${root_path}:${PYTHONPATH}

python pgm_feature_generation.py --yml_cfg_file ${bsn_cfg_file}
python pem_train.py --yml_cfg_file ${bsn_cfg_file}
python pem_test.py --yml_cfg_file ${bsn_cfg_file}
python post_processing.py --yml_cfg_file ${bsn_cfg_file}
echo "Evaluating..."
python ../../evaluate/anet_eval_proposals.py --yml_cfg_file ${bsn_cfg_file}