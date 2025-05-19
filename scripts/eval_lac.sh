# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python lac.py \
--input_model $1 \
--rotated_matrix_path $2 \
--num_hidden_layers -1 \
--device 0 \
--model_max_length 2048 \
$( [ "$3" = "false" ] && echo "--no-bf16" ) \
--zero_shot \
--w_bits $5 \
--a_bits $6 \
--k_bits $7 \
--v_bits $7 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--no-w_rtn \
--lac \
--cali_epochs 5 \
--cali_bsz $4 \
--cali_lr 0.05 \
--k_groupsize 128 \
--v_groupsize 128 \
--log_dir "./" \
--exp_name "lac" \


