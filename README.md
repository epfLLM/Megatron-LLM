
<div align="center">
  <img src="docs/imgs/llama-falcon.png"  width="500">
</div>

# Megatron-LLM

This library enables pre-training and fine-tuning of large language models (LLMs) at scale.
Our repository is a modification of the [original Megatron-LM codebase](https://github.com/NVIDIA/Megatron-LM) by Nvidia.

Added key features include:
- [LLaMa](https://arxiv.org/abs/2302.13971), [LLaMa 2](https://arxiv.org/abs/2307.09288), [Falcon](https://huggingface.co/tiiuae), and [Code Llama](https://arxiv.org/abs/2308.12950) support
- Support training of large models (70B Llama 2, 65B Llama 1 and 40B Falcon) on commodity hardware on multiple nodes.
- 3-way parallelism: tensor parallel, pipeline parallel and data parallel training (inherited from Megatron)
- grouped-query attention (GQA) and multi-query attention (MQA)
- Rotary Position Embeddings (RoPE) [was added independently by the Megatron project subsequent to us]
- RMS layer norm
- FlashAttention 2
- BF16 / FP16 training
- Support for special tokens & tokenizers
- WandB integration
- ROtary Position Embedding ([ROPE](https://together.ai/blog/llama-2-7b-32k)) scaling for extra large context windows
- Support for publishing your checkpoints to the huggingface hub.
- Instruction finetuning support
- Metrics support: Ease to add custom metrics to evaluate on the validation set while training

# Documentation

Take a look at [the online documentation](https://epfllm.github.io/Megatron-LLM).

Alternatively, build the docs from source:
```
cd docs/
pip install -r requirements.txt
make html
```


# Citation

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
  url          = {https://github.com/epfLLM/Megatron-LLM}
}
</pre>
