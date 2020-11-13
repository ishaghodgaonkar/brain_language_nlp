# ECE 570 Final Project README
--------------------------------------------------------------------------------------------------------------------------------------------
Part 1: Brain data prediction

The changes made in the brain_language_nlp repository (in brain_language_nlp folder) were in the following files: extract_nlp_features.py and xl_net_utils.py. The changed code is a few lines (10-15) total. xl_net_utils.py is a copy of xl_utils.py which extracts features from Transformer-XL, with a few changes to run for XLNet instead.

To extract features, build the encoding model, and then generate predictions for BERT for all layers for subject F, run the following command on a server which can run continuously for a day or two without being interrupted:
```
nohup python3 plot_graphs_paper.py
disown
```

To do the same for XLNet:
```
nohup python3 plot_graphs_paper_xl_net.py
disown
```

Then run 

```
python3 extract_nums.py
```

to get the final numbers to plot by averaging each list obtained.

To then plot the graphs using these numbers:


```
python3 draw_plots.py
```

The final .txt files are included in the submission for convenience to run the last step and obtain the results shown in my term paper.

Models were obtained from Pytorch Transformers library and data was obtained from original repository (link: https://github.com/mtoneva/brain_language_nlp)

--------------------------------------------------------------------------------------------------------------------------------------------
Part 2: NLP task performance evaluation

For the text classification + semantic analysis tasks, I built off of 2 scripts published in this article: https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b


Both scripts are combined together in Text Classification BERT and XL_net and each cell is marked with "NEW CODE" or "MODIFIED CODE" where I introduced new code or modified existing code. I introduced a new dataset from Kaggle (a COVID-19 tweets dataset) and XLnet. Now both BERT and XLnet can be trained on the existing News dataset already implemented or the COVID tweets dataset. To do this, run all cells until Preliminaries, and in the first cell under Preliminaries, change the model to BERT or XLNet and the dataset to NEWS or COVID. Then run the rest of the code cells to produce the graphs.


# END ECE 570 Final Project README



# Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain)

This repository contains code for the paper [Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain)](https://arxiv.org/pdf/1905.11833.pdf)

Bibtex: 
```
@inproceedings{brain_language_nlp,
    title={Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain)},
    author={Toneva, Mariya and Wehbe, Leila},
    booktitle={NeurIPS},
    year={2019}
}
```
## fMRI Recordings of 8 Subjects Reading Harry Potter
You can download the already [preprocessed data here](https://drive.google.com/drive/folders/1Q6zVCAJtKuLOh-zWpkS3lH8LBvHcEOE8?usp=sharing). This data contains fMRI recordings for 8 subjects reading one chapter of Harry Potter. The data been detrended, smoothed, and trimmed to remove the first 20TRs and the last 15TRs. For more information about the data, refer to the paper. We have also provided the precomputed voxel neighborhoods that we have used to compute the searchlight classification accuracies. 

The following code expects that these directories are positioned under the data folder in this repository (e.g. `./data/fMRI/` and `./data/voxel_neighborhoods`.


## Measuring Alignment Between Brain Recordings and NLP representations

Our approach consists of three main steps:
1. Derive representations of text from an NLP model
2. Build an encoding model that takes the derived NLP representations as input and predicts brain recordings of people reading the same text
3. Evaluates the predictions of the encoding model using a classification task

In our paper, we present alignment results from 4 different NLP models - ELMo, BERT, Transformer-XL, and USE. Below we provide an overview of how to run all three steps.


### Deriving representations of text from an NLP model

Needed dependencies for each model:
- USE: Tensorflow < 1.8,  `pip install tensorflow_hub`
- ELMo: `pip install allennlp`
- BERT/Transformer-XL: `pip install pytorch_pretrained_bert`


The following command can be used to derive the NLP features that we used to obtain the results in Figures 2 and 3:
```
python extract_nlp_features.py
    --nlp_model [bert/transformer_xl/elmo/use]   
    --sequence_length s
    --output_dir nlp_features
```
where s ranges from to 1 to 40. This command derives the representation for all sequences of `s` consecutive words in the stimuli text in `/data/stimuli_words.npy` from the model specified in `--nlp_model` and saves one file for each layer in the model in the specified `--output_dir`. The names of the saved files contain the argument values that were used to generate them. The output files are numpy arrays of size `n_words x n_dimensions`, where `n_words` is the number of words in the stimulus text and `n_dimensions` is the number of dimensions in the embeddings of the specified model in `--nlp_model`. Each row of the output file contains the representation of the most recent `s` consecutive words in the stimulus text (i.e. row `i` of the output file is derived by passing words `i-s+1` to `i` through the pretrained NLP model).


### Building encoding model to predict fMRI recordings

Note: This code has been tested using python3.7

```
python predict_brain_from_nlp.py
    --subject [F,H,I,J,K,L,M,N]
    --nlp_feat_type [bert/elmo/transformer_xl/use]   
    --nlp_feat_dir INPUT_FEAT_DIR
    --layer l
    --sequence_length s
    --output_dir OUTPUT_DIR
```

This call builds encoding models to predict the fMRI recordings using representations of the text stimuli derived from NLP models in step 1 above (`INPUT_FEAT_DIR` is set to the same directory where the NLP features from step 1 were saved, `l` and `s` are the layer and sequence length to be used to load the extracted NLP representations). The encoding model is trained using ridge regression and 4-fold cross validation. The predictions of the encoding model for the heldout data in every fold are saved in an output file in the specified directory `OUTPUT_DIR`. The output filename is in the following format: `predict_{}_with_{}_layer_{}_len_{}.npy`, where the first field is specified by `--subject`, the second by `--nlp_feat_type`, and the rest by `--layer` and `--sequence_length`.

### Evaluating the predictions of the encoding model using classification accuracy

Note: This code has been tested using python3.7

```
python evaluate_brain_predictions.py
    --input_path INPUT_PATH
    --output_path OUTPUT_PATH
    --subject [F,H,I,J,K,L,M,N]
```

This call computes the mean 20v20 classification accuracy (over 1000 samplings of 20 words) for each encoding model (from each of the 4 CV folds). The output is a `pickle` file that contains a list with 4 elements -- one for each CV fold. Each of these 4 elements is another list, which contains the accuracies for all voxels. `INPUT_PATH` is the full path (including the file name) to the predictions saved in step 2 above. `OUTPUT_PATH` is the complete path (including file name) to where the accuracies should be saved. 

The following extracts the average accuracy across CV folds for a particular subject:
```
import pickle as pk
import numpy as np
loaded = pk.load(open('{}_accs.pkl'.format(OUTPUT_PATH), 'rb'))
mean_subj_acc_across_folds = loaded.mean(0)
```
