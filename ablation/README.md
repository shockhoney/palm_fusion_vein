# Ablation training

Run commands from palm_fusion_vein. All ablation logic is contained in this directory; the original model and training files are unchanged.

## 1. Teacher fusion

Variants:

- concat_linear: concatenate the two unimodal embeddings, then apply a linear projection.
- cross_attention: cross-modal attention, concatenation, and linear projection.
- channel_adaptive: channel-adaptive fusion without cross-modal attention.
- full: cross-modal attention followed by channel-adaptive fusion.

Reuse the existing unimodal checkpoints:

~~~powershell
python ablation\teacher_fusion_ablation.py --mode train --variant all --skip_stage1 --palm_ckpt <PALM_STAGE1_CHECKPOINT> --vein_ckpt <VEIN_STAGE1_CHECKPOINT> --phase2_train data_txt\<DATASET>_phase2_train.txt --phase2_val data_txt\<DATASET>_phase2_val.txt --save_dir outputs\ablation\teacher_fusion\<DATASET> --seed 42
~~~

Use the matching Tongji or CUMT palm/vein checkpoints for each seed. Repeat with seeds 42, 44, and 46.

Test one trained variant with its matching checkpoint:

~~~powershell
python ablation\teacher_fusion_ablation.py --mode test --variant cross_attention --ckpt outputs\ablation\teacher_fusion\<DATASET>\teacher_fusion_cross_attention_seed42\stage2_best.pth --palm_list data_txt\<DATASET>_palmprint_list.txt --vein_list data_txt\<DATASET>_palmvein_list.txt --pair_txt data_txt\<DATASET>_phase2_test.txt --out_csv outputs\ablation\teacher_fusion\<DATASET>\cross_attention_seed42.csv
~~~

## 2. Distillation loss

| Variant | Embedding KD | Relational KD | Confidence | Ramp-up |
|---|---:|---:|---:|---:|
| cls | No | No | No | No |
| emb | Yes | No | No | Yes |
| rel | No | Yes | No | Yes |
| emb_rel | Yes | Yes | No | Yes |
| full_no_ramp | Yes | Yes | Yes | No |
| full | Yes | Yes | Yes | Yes |

~~~powershell
python ablation\distillation_ablation.py --mode train --variant all --train_list data_txt\<DATASET>_phase2_train.txt --val_list data_txt\<DATASET>_phase2_val.txt --teacher_ckpt <FULL_TEACHER_CHECKPOINT> --save_dir outputs\ablation\distillation\<DATASET> --seed 42
~~~

Repeat with the full teacher checkpoint from the same dataset and seed.

~~~powershell
python ablation\distillation_ablation.py --mode test --variant emb_rel --ckpt outputs\ablation\distillation\<DATASET>\distill_emb_rel_seed42\student_best_distill.pth --palm_list data_txt\<DATASET>_palmprint_list.txt --vein_list data_txt\<DATASET>_palmvein_list.txt --pair_txt data_txt\<DATASET>_phase2_test.txt --out_csv outputs\ablation\distillation\<DATASET>\emb_rel_seed42.csv
~~~

## 3. Student architecture

- mobile_concat: MobileFaceNet without ECA + concat/linear fusion.
- mobile_eca_concat: MobileFaceNet+ECA + concat/linear fusion.
- mobile_gate: MobileFaceNet without ECA + bottleneck-gated fusion.
- mobile_eca_gate: MobileFaceNet+ECA + bottleneck-gated fusion.

Keep the distillation objective fixed:

~~~powershell
python ablation\student_architecture_ablation.py --mode train --variant all --train_list data_txt\<DATASET>_phase2_train.txt --val_list data_txt\<DATASET>_phase2_val.txt --teacher_ckpt <FULL_TEACHER_CHECKPOINT> --save_dir outputs\ablation\student_architecture\<DATASET> --seed 42
~~~

~~~powershell
python ablation\student_architecture_ablation.py --mode test --variant mobile_gate --ckpt outputs\ablation\student_architecture\<DATASET>\student_arch_mobile_gate_seed42\student_best_distill.pth --palm_list data_txt\<DATASET>_palmprint_list.txt --vein_list data_txt\<DATASET>_palmvein_list.txt --pair_txt data_txt\<DATASET>_phase2_test.txt --out_csv outputs\ablation\student_architecture\<DATASET>\mobile_gate_seed42.csv
~~~
