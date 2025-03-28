#!/bin/bash

# Define the arguments
i="1"
nn="64"
nl="4"
o="1"
tr="1"
ep="400"
epr="400"
b="50000"
kl="0.0"
pr="0.0003"
dr="0.01"
ns="100"
ne="100"
s="36"

# Run the Python script with the arguments
###############################################################################
#                       BNN
###############################################################################
  
python post_process.py \
  --architecture "BNN" \
  --input_dim $i \
  --n_neurons $nn \
  --n_layers $nl \
  --output_dim $o \
  --tr_layers $tr \
  --n_epochs $ep \
  --n_epochs_pretrain $epr \
  --batch_size $b \
  --dropout_prob $dr \
  --n_samples $ns\
  --n_ensembles $ne\
  --kl_weight $kl \
  --prior_std $pr \
  --seed $s
  
###############################################################################
#                       BNN_NLL
###############################################################################
python post_process.py \
  --architecture "BNN_NLL" \
  --input_dim $i \
  --n_neurons $nn \
  --n_layers $nl \
  --output_dim $o \
  --tr_layers $tr \
  --n_epochs $ep \
  --n_epochs_pretrain $epr \
  --batch_size $b \
  --dropout_prob $dr \
  --n_samples $ns\
  --n_ensembles $ne\
  --kl_weight $kl \
  --prior_std $pr \
  --seed $s
  
###############################################################################
#                       MCD
###############################################################################
python post_process.py \
  --architecture "MCD" \
  --input_dim $i \
  --n_neurons $nn \
  --n_layers $nl \
  --output_dim $o \
  --tr_layers $tr \
  --n_epochs $ep \
  --n_epochs_pretrain $epr \
  --batch_size $b \
  --dropout_prob $dr \
  --n_samples $ns\
  --n_ensembles $ne\
  --kl_weight $kl \
  --prior_std $pr \
  --seed $s
  
###############################################################################
#                       MCD_NLL
###############################################################################

python post_process.py \
  --architecture "MCD_NLL" \
  --input_dim $i \
  --n_neurons $nn \
  --n_layers $nl \
  --output_dim $o \
  --tr_layers $tr \
  --n_epochs $ep \
  --n_epochs_pretrain $epr \
  --batch_size $b \
  --dropout_prob $dr \
  --n_samples $ns\
  --n_ensembles $ne\
  --kl_weight $kl \
  --prior_std $pr \
  --seed 42
  
###############################################################################
#                       DEN
###############################################################################
python post_process.py \
  --architecture "DEN" \
  --input_dim $i \
  --n_neurons $nn \
  --n_layers $nl \
  --output_dim $o \
  --tr_layers $tr \
  --n_epochs $ep \
  --n_epochs_pretrain $epr \
  --batch_size $b \
  --dropout_prob $dr \
  --n_samples $ns\
  --n_ensembles $ne\
  --kl_weight $kl \
  --prior_std $pr \
  --seed $s

###############################################################################
#                       DEN MCD
###############################################################################

ne="10"

python post_process.py \
  --architecture "DEN_MCD" \
  --input_dim $i \
  --n_neurons $nn \
  --n_layers $nl \
  --output_dim $o \
  --tr_layers $tr \
  --n_epochs $ep \
  --n_epochs_pretrain $epr \
  --batch_size $b \
  --dropout_prob $dr \
  --n_samples $ns\
  --n_ensembles $ne\
  --kl_weight $kl \
  --prior_std $pr \
  --seed $s
