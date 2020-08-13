## Preprocessing component

### Data Paths
Paths for I/O

argument | description | default value
---- | ---- | ----
-transcript_path | raw transcript | ../raw_data/transcript/ 
-summary_path | human written summary | ../raw_data/summary/ 
-index_path | index path for re-ordering | ../ext_data/index/ 
-story_path | output path of story mode | ../ext_data/story/ 
-txt_path | new transcript input path | ../txt_data/ 
-txt_output_path | output path of txt mode | ../ext_data/text/

### Mode Settings
Choose the mode to run. Story mode is used for a labeled dataset and txt mode is used for new raw transcript input

argument | description | default value
---- | ---- | ----
-mode | story/txt | story 
-split | whether to do train/test split | True 
-seed | random seed for split | 

### Algorithm Settings
Customize the lsa and clustering algorithm by changing the following argument

argument | description | default value
---- | ---- | ----
-n_gram | lower and upper bound of n grams in LSA | (1,1)
-lsa_num | feature dimension for lsa | 30
-algorithm | equal size clustering or kmeans | ec
-sent_num | average sentence num for each cluster | 10
-min_words | min non-stopwords sentence for clustering | 3
 
