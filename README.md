
<div align="center">
  <img src="imgs/llama-falcon.png"  width="500">
</div>

# Megatron-LLM

This library enables pre-training and fine-tuning of large language models (LLMs) at scale.
Our repository is a modification of the [original Megatron-LM codebase](https://github.com/NVIDIA/Megatron-LM) by Nvidia.

Added key features include:
- [Llama](https://arxiv.org/abs/2302.13971), [Llama 2](https://arxiv.org/abs/2307.09288) and [Falcon](https://huggingface.co/tiiuae) support
- support training of large models (70B Llama2, 65B Llama1 and 40B Falcon) on commodity hardware on multiple nodes
- 3-way parallelism: tensor parallel, pipeline parallel and data parallel training (inherited from Megatron)
- grouped-query attention (GQA) and multi-query attention (MQA)
- Rotary Position Embeddings (RoPE) [was added independently by the Megatron project subsequent to us]
- RMS layer norm
- FlashAttention 2
- BF16 / FP16 training
- Support for special tokens & tokenizers
- WandB integration

# Setup

Because of heavy use of [Apex](https://github.com/NVIDIA/apex), this codebase is currently for Nvidia GPUs only.

Like [Megatron](https://github.com/NVIDIA/Megatron-LM), we recommend [the NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). Instructions for obtaining and running this is at the link above.

A C++ compiler and the ninja build system may also be necessary.

We additionally add a dependency on [HuggingFace Transfomers](https://pypi.org/project/transformers/). `einops` is also required. 

PyTorch>=2.0.0 is required for flash attention.

A recommended entrypoint is `examples/finetune.sh`.
Information on preparing data is at `tokenize-utils/README.md`.


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
