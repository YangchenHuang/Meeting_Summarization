## Extracitve component

### Preprocessing

#### Data Paths
Paths for I/O

argument | description | default value
---- | ---- | ----
-story_path | story file for each topic | ../ext_data/story/
-index_path | index path for re-ordering | ../ext_data/index/
-json_path | temp path for intermediate processing | ../ext_data/json_data/
-bert_path | preprocessed torch format data for training | ../ext_data/bert_data/

#### Mode Settings
Customize some preprocessing logics

argument | description | default value
---- | ---- | ----
-mode | get distribution or one hot on rouge | sent_dist
-min_src_ntokens_per_sent | min tokens per sentence | ../ext_data/index/
-max_src_ntokens_per_sent | max tokens per sentence | ../ext_data/json_data/

### Main Body
#### Data Paths
Paths for I/O

argument | description | default value
---- | ---- | ----
-bert_data_path | preprocessed torch format data for training | ../ext_data/bert_data/
-model_path | saved model checkpoint | ../models/extract/ 
-result_path | output result extractive summary | ../ext_data/summary/
-story_path | output result story for abstractive input | ../ext_data/result_story/
-temp_dir | temp path for pretrained models | ../temp/
-train_from | train from saved checkpoint | 
-test_from | test from saved checkpoint | 
-log_file | log file for debugging | ../logs/extractive.log


#### Mode Settings
Choose the modeling and running mode

argument | description | default value
---- | ---- | ----
-mode | train/validate/test | train
-test_txt | for testing new meeting transcript | False
-world_size | gpu world size, 0 if cpu | 1
-visible_gpus | visible gpu ids, automatically generate if no input | 
-gpu_ranks | gpu ranks, automatically generate if no input | 


#### Hyperparameters
Model and optimizer hyperparameters

argument | description | default value
---- | ---- | ----
-use_interval | whether to use the interval sentence feature | False
-ff_size | feed forward network hidden size | 512
-heads | heads for multi-head attention | 16
-inter_layers | transformer layers for summarization | 16
-dropout | dropout rate | 0.1
-optim | optimizer chosen | adam
-lr | learning rate | 2e-7
-beta1 | beta1 for adam | 0.9
-beta2 | beta2 for adam | 0.999
-max_grad_norm | clip gradient norm, 0 if not | 0
-decay_step | step starting learning rate decay | 1000

#### Training Control
Arguments that control the whole training procedure

argument | description | default value
---- | ---- | ----
-batch size | batch size for training | 1000
-test_batch_size | batch size for testing | 20000
-save_checkpoint_steps | how often to save checkpoints | 1000
-accum_count | how often to accumulate gradient | 5
-report_every | how often to report result | 50
-train_steps | total epochs for training | 5000
-seed | random seed for training |
