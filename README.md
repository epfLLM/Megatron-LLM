This library enables pre-training and fine-tuning of large language models (LLMs) at scale.
Our repository is a modification of the [original Megatron-LM codebase](https://github.com/NVIDIA/Megatron-LM) by Nvidia.

Added key features include:
- Llama, Llama 2 and Falcon support
- support training of large models (70B Llama2, 65B Llama1 and 40B Falcon) on commodity hardware on multiple nodes
- 3-way parallelism: tensor parallel, pipeline parallel and data parallel training (inherited from Megatron)
- grouped-query attention (GQA) and multi-query attention (MQA)
- Rotary Position Embeddings (RoPE)
- RMS layer norm
- FlashAttention 2
- BF16 / FP16 training
- Support for special tokens & tokenizers

If you use this software please cite it:
<pre>
@software{epfmgtrn,
  author       = {Alejandro Hernández Cano  and
                  Matteo Pagliardini  and
                  Kyle Matoba  and
                  Amirkeivan Mohtashami  and
                  Olivia Simin Fan  and
                  Axel Marmet  and
                  Deniz Bayazit  and
                  Igor Krawczuk  and
                  Zeming Chen  and
                  Francesco Salvi  and
                  Antoine Bosselut  and
                  Martin Jaggi},
  title        = {epfLLM Megatron-LM},
  year         = 2023,
  url          = {https://github.com/epfLLM/Megatron-LM}
}
</pre>
