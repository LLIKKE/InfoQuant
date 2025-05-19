# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python get_rotation_matrix.py \
--input_model $1 \
--model_max_length 2048 \
--num_hidden_layers -1 \
--device 0 \
--w_bits $3 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--no-lac \
--w_rtn \
--block_diag $2 \
--aug_token 30 \
--lrs 2 2 2 \
--epochs 15 \
--batch_size 4 \
--T 2 \
--log_dir "./" \
--exp_name "rotated_matrix" \


