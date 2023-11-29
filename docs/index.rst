Welcome to Megatron-LLM's documentation!
========================================

.. image:: imgs/llama-falcon.png

The `Megatron-LLM <https://github.com/epfLLM/Megatron-LLM/>`_ library enables pre-training and fine-tuning of large language models (LLMs) at scale.
Our repository is a modification of the `original Megatron-LM codebase <https://github.com/NVIDIA/Megatron-LM>`_ by Nvidia.

Added key features include:

- architectures supported: `LLaMa <https://arxiv.org/abs/2302.13971>`_, `LLaMa 2 <https://arxiv.org/abs/2307.09288>`_, `Falcon <https://huggingface.co/tiiuae>`_, `Code Llama <https://together.ai/blog/llama-2-7b-32k>`_ and `Mistral https://arxiv.org/abs/2310.06825`_.
- support training of large models (70B Llama 2, 65B Llama 1, 34B Code Llama, 40B Falcon and Mistral) on commodity hardware on multiple nodes
- 3-way parallelism: tensor parallel, pipeline parallel and data parallel training (inherited from Megatron)
- full pretraining, finetuning and instruct tuning support
- Support for special tokens & tokenizers
- grouped-query attention (GQA) and multi-query attention (MQA)
- Rotary Position Embeddings (RoPE), RMS layer norm, Lima dropout
- `RoPE scaling <https://together.ai/blog/llama-2-7b-32k>`_ for longer attention context support
- FlashAttention 2
- BF16 / FP16 training
- WandB integration
- Metrics support: Ease to add custom metrics to evaluate on the validation set while training
- Conversion to and from Hugging Face hub

Example models trained with `Megatron-LLM <https://github.com/epfLLM/Megatron-LLM/>`_: See `README <https://github.com/epfLLM/Megatron-LLM/>`_.

User guide
----------

For information on installation and usage, take a look at our user guide.

.. toctree::
   :maxdepth: 2

   guide/index


API
---

Detailed information about Megatron-LLM components:

.. toctree::
   :maxdepth: 2

   api/index




Citation
--------

If you use this software please cite it:

.. code-block:: bib

   @software{epfmgtrn,
     author       = {Alejandro Hernández Cano  and
                     Matteo Pagliardini  and
                     Andreas Köpf  and
                     Kyle Matoba  and
                     Amirkeivan Mohtashami  and
                     Xingyao Wang  and
                     Olivia Simin Fan  and
                     Axel Marmet  and
                     Deniz Bayazit  and
                     Igor Krawczuk  and
                     Zeming Chen  and
                     Francesco Salvi  and
                     Antoine Bosselut  and
                     Martin Jaggi},
     title        = {epfLLM Megatron-LLM},
     year         = 2023,
     url          = {https://github.com/epfLLM/Megatron-LLM}
   }
