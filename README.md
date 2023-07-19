# Big-GCVAE
Big GCVAE: Probabilistic Graphical BERT-GPT2 Transformer Model for Failure Analysis Triplet Generation
![Complete flow](https://github.com/FA4-0/Big-GCVAE/blob/main/complete_flow.png)

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
![Complete flow](https://github.com/FA4-0/Big-GCVAE/blob/main/biggcvae.png)
