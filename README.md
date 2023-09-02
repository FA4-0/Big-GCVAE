# Big-GCVAE
### Big GCVAE: Probabilistic Graphical BERT-GPT2 Transformer Model for Failure Analysis Triplet Generation

![Complete worflow](https://github.com/FA4-0/Big-GCVAE/blob/main/complete_flow.png)

------------------------------

## Abstract

```
Pre-trained Language Models (LMs) have gained significant attention in the field of Natural Language
Processing (NLP), particularly for tasks such as text summarization, generation, and question answering.
The success of LMs can be attributed to the attention mechanism introduced in Transformer models,
which have surpassed the performance of traditional Recurrent Neural Network Models (e.g., LSTM)
in modeling sequential data. In this paper, we harness the power of the attention mechanism in pre-trained
causal language models for the downstream task of Failure Analysis Triplet Generation (FATG), which
 involves generating a sequence of text-encoded steps for analyzing  defective components in the
semiconductor industry. In particular, we perform extensive comparative analysis of various transformer
 models for the FATG task is performed to find that BERT-GPT-2 Transformer (Big GCVAE),
 finetuned on a proposed Generalized-Controllable Variational AutoEncoder loss (GCVAE) exhibits
superior performance compared to other transformer models. Specifically, we observe that fine-tuning
 the Transformer style BERT-GPT2 on the GCVAE loss yields a smooth and
interpretable latent representation with high quality generative performance across all scoring
metrics.
```
![Big GCVAE](https://github.com/FA4-0/Big-GCVAE/blob/main/biggcvae.png)


------------------------------

## How to use

- Clone repo: ```git clone https://github.com/FA4-0/Big-GCVAE ```
- Run ```training``` and ```evaluation``` on example data either using [Bash Slurm file](https://github.com/FA4-0/Big-GCVAE/blob/main/slurm/pretrainer_gcvae.job) on HCP or:
  ```python
  python pretrainer_gcvae.py \
        --model_type_enc bert-base-uncased \
        --model_name_or_path_enc bert-base-uncased \
        --model_type_dec gpt2-medium \
        --model_name_or_path_dec gpt2-medium \
        --do_eval \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 1 \
        --learning_rate 5e-5 \
        --num_train_epochs 1.0 \
        --output_dir result/ \
        --eval_dir evaluation/ \
        --overwrite_output_dir \
	    --length_weighted_loss \
        --fp16 \
        --fp16_opt_level O2 \
        --gradient_accumulation_steps 1 \
        --seed 42 \
        --do_lower_case \
        --encoder_decoder_sep \
        --warmup_steps 100 \
        --logging_steps 100 \
        --save_steps 100 \
        --evaluate_during_training \
        --adam_epsilon 1e-8 \
        --weight_decay 0.05 \
        --max_grad_norm 1.0 \
        --return_token_type_ids \
        --dim_target_kl 1.0 \
        --ratio_zero .5 \
        --ratio_increase .25 \
       	--temperature 1.9 \
       	--top_p 0.95 \
       	--top_k 10 \
       	--num_interpolation_steps 1 \
       	--use_variational_loss \
       	--vae_model_name gcvae \
       	--mmd_type mmd \
        --beta 5.0 \
        --gamma 500.0 \
        --init_kld 1 \
        --init_bce 0.01 \
        --init_mmd 0.01 \
        --max_steps -1
  ```

## Results
- ```Evaluation``` loss:

| Model           | Evaluation | Reconstruction | KL divergence  |
|-----------------|------------|----------------|----------------|
|                 |  **loss**      |  **loss**          |                |
| Big VAE          | **1.10**  | 128.34  | 6.49 |
|    Big ControlVAE      | 1.18  | 1.10  | 9.85 |
| Big GCVAE†  | 1.18  | 1.09  | 8.23 |
|Big GCVAE‡ | 1.11  | **1.09**  | 3.80 |

	Performance evaluation of Big GCVAE models and 	its derivatives. Both Big GCVAE† (Maximum Mean Discrepancy) and 
 	Big GCVAE‡ (Squared Mahalanobis) have the lowest reconstruction loss compared to Big VAE (Li et al., 2020).

- ```Evaluation metrics```:
  
| Model           | BLEU-1 | BLEU-3 | MET.  | ROUGE-1  | ROUGE-L  | LESE-1   | LESE-3   |
|-----------------|--------|--------|-------|----------|----------|----------|----------|
|                 |        |        |       | **F1-score** | **F1-score** | **F1-score** | **F1-score** |
| GPT2-M          | 21.26  | 15.47  | 26.74 | 29.15    | 26.56    | 19.41    | 8.55     |
| Big VAE         | 21.87  | 16.15  | 26.64 | 31.23    | 28.73    | 21.46    | 9.57     |
| Big ControlVAE  | 22.25  | 16.38  | 27.10 | 31.55    | 28.89    | 21.54    | 9.58     |
| Big GCVAE† | 22.09  | 16.25  | 27.01 | 31.17    | 28.58    | 21.36    | 9.50     |
| Big GCVAE‡ | **22.53**  | **17.00**  | **27.63** | **31.71**    | **29.08**    | **21.79**    | **9.70**    |
	
 	Model comparison on BLEU (Papineni et al., 2002), ROUGE-1 & L (Lin, 2004), METEOR (MET.) (Banerjee
	and Lavie, 2005) and LESE (Ezukwoke et al., 2022b). Higher values (in bold-blue) is preferred for 
 	all metric except Lev-n	(average n-gram Levenshstein distance). Big GCVAE‡ performs better across
  	all evaluation metric. Observe the approximately 3-point increase in performance of the generative
   	strength for ROUGE-1 and LESE-1 and a comparable increase for the triplet evaluations.
 	
- ```Latent representation```
  
![Latent representation](https://github.com/FA4-0/Big-GCVAE/blob/main/latent_s.png)
  
	2D Latent representation space (top) and t-SNE Embedding (bottom). Observe the quality of clusters in the latent
	space for Big GCVAE† (best), Big VAE (second best) and Big GCVAE‡ (less fuzzy). The latent space of Big ControlVAE
	is the most fuzzy with overlapping cluster of densities in t-SNE Embedding space.

## Citation
- ```Under Review``` @ ```Journal of Intelligent Manufacturing (JIM)```


