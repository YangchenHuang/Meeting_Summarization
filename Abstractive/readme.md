## Abstractive component

In this part, the extracitve summary is fed into an encoder layer (BERT or Longformer) to obtain word representation.
Taking encoder output and decoder input, the decoder together with the predictor layer are trained to predict the 
probability distribution of each word in the vocabulary for each position of the output sequence. When doing test,
a start token is used as the first decoder input and the model will use the output probability distribution to conduct 
beam search to obtain predicted sequence. When end token is predicted, the model will output the final abstractive 
summary.

### Preprocessing

#### Data Paths
Paths for I/O

argument | description | default value
---- | ---- | ----
-story_path | extractive summary input | ../ext_data/result_story/
-json_path | temp path for intermediate processing | ../abs_data/json_data/
-bert_path | torch data using bert tokenizer | ../abs_data/bert_data/
-long_path | torch data using longformer tokenizer | ../abs_data/long_data/

#### Mode Settings
Choose the tokenizer to use and corresponding preprocessing logics

argument | description | default value
---- | ---- | ----
-tokenizer | bert/longformer | bert
-shard_size | shard size for formating training set | 2000
-min_src_ntokens_per_sent | min tokens per sentence | 5
-max_src_ntokens_per_sent | max tokens per sentence | 200
-min_src_nsents | min sentences for training data point | 3
-max_src_nsents | max sentences for training data point | 300

### Main Body
#### Data Paths
Paths for I/O

argument | description | default value
---- | ---- | ----
-data_path | preprocessed torch format data for training | ../abs_data/bert_data/
-model_path | saved model checkpoint | ../models/abstract/
-result_path | output result as single file | ../abs_data/result/
-summary_path | final summary for each raw transcript | ../result_summary/
-temp_dir | temp path for pretrained models | ../temp/
-train_from | train from saved checkpoint | 
-test_from | test from saved checkpoint | 
-log_file | log file for debugging | ../logs/abstractive.log


#### Mode Settings
Choose the modeling and running mode

argument | description | default value
---- | ---- | ----
-mode | train/validate/test | train
-test_txt | for testing new meeting transcript | False
-validate_rouge | whether to validate rouge score | False
-encoder | encoder used for training (bert/longformer) | bert
-world_size | gpu world size, 0 if cpu | 1
-visible_gpus | visible gpu ids, automatically generate if no input | 
-gpu_ranks | gpu ranks, automatically generate if no input | 


#### Hyperparameters
Model and optimizer hyperparameters

argument | description | default value
---- | ---- | ----
-max_pos | max tokens to input | 512
-large | whether to use large pretrained models | False
-finetune | whether to finetue encoder | 512
-use_bert_emb | whether to use bert embedding for decoder | True
-dec_ff_size | feed forward network hidden size | 2048
-dec_heads | heads for multi-head attention | 8
-dec_inter_layers | transformer layers for decoder | 6
-dec_hidden_size | hidden for decoder | 768
-dec_dropout | dropout rate | 0.2
-label_smoothing | label smoothing when training, 0 if not | 0.1
-generator_shard_size | shard size for generator layer | 32
-optim | optimizer chosen | adam
-sep_optim |whether to optimize encoder and decoder separately
-lr | joint learning rate | 2e-3
-lr_enc | learning rate for encoder | 2e-3
-lr_dec | learning rate for decoder | 2e-3
-beta1 | beta1 for adam | 0.9
-beta2 | beta2 for adam | 0.999
-max_grad_norm | clip gradient norm, 0 if not | 0
-warmup_steps | warmup steps for joint optimization | 500
-warmup_steps_enc | warmup steps for encoder optimization | 500
-warmup_steps_dec | warmup steps for decoder optimization | 500

#### Predictor Settings
Customize the predictor

argument | description | default value
---- | ---- | ----
-alpha | penalty for long sequences | 0.1
-beam_size | size of beam search | 10
-min_length | min tokens for output summary | 280
-max_length | max tokens for output summary | 400
-block_trigram | whether to block same trigram in summary | True


#### Training Control
Arguments that control the whole training procedure

argument | description | default value
---- | ---- | ----
-batch size | batch size for training | 200
-test_batch_size | batch size for testing | 200
-save_checkpoint_steps | how often to save checkpoints | 500
-accum_count | how often to accumulate gradient | 5
-report_every | how often to report result | 50
-train_steps | total epochs for training | 2000
-seed | random seed for training |
