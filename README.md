# Don't Judge a Language Model by Its Last Layer: Contrastive Learning with Layer-Wise Attention Pooling


Paper link: https://aclanthology.org/2022.coling-1.405/

To be published in [**Coling 2022**](https://coling2022.org/)

Our code is mainly based on the code of [SimCSE](https://arxiv.org/abs/2104.08821). Please refer to their [repository](https://github.com/princeton-nlp/SimCSE) for more detailed information.

### Requirements
* Python 3.8

### Install other packages
```
pip install -r requirements.txt
```

### Download the pretraining dataset
```
cd data
bash download_nli.sh
```

### Download the downstream dataset
```
cd SentEval/data/downstream/
bash download_dataset.sh
```

## Training
(Using Multi-GPU `run_sup_layerattnpooler.sh`.)
```bash
python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/nli_for_simcse.csv \
    --output_dir result/bert-base-uncased-cl-layerattnpooler \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --learning_rate 2e-5 \
    --max_seq_length 64 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 100 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
```

## Citations

Please cite our paper if they are helpful to your work!

```bibtex
@inproceedings{oh2022don,
  title={Don’t Judge a Language Model by Its Last Layer: Contrastive Learning with Layer-Wise Attention Pooling},
  author={Oh, Dongsuk and Kim, Yejin and Lee, Hodong and Huang, H Howie and Lim, Heui-Seok},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  pages={4585--4592},
  year={2022}
}
```
