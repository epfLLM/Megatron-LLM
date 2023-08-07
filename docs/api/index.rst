API
===

megatron
--------

.. autosummary::
   :toctree: megatron

   megatron.arguments
   megatron.checkpointing
   megatron.dist_signal_handler
   megatron.global_vars
   megatron.indexer
   megatron.initialize
   megatron.memory
   megatron.microbatches
   megatron.optimizer_param_scheduler
   megatron.p2p_communication
   megatron.schedules
   megatron.text_generation_server
   megatron.timers
   megatron.training
   megatron.utils
   megatron.wandb_logger

megatron.core
-------------

.. autosummary::
   :toctree: megatron/core

   megatron.core.parallel_state
   megatron.core.utils


megatron.core.tensor_parallel
-----------------------------

.. autosummary::
   :toctree: megatron/core/tensor_parallel

   megatron.core.tensor_parallel.cross_entropy
   megatron.core.tensor_parallel.data
   megatron.core.tensor_parallel.layers
   megatron.core.tensor_parallel.mappings
   megatron.core.tensor_parallel.random
   megatron.core.tensor_parallel.utils

megatron.data
-------------

.. autosummary::
   :toctree: megatron/data

   megatron.data.autoaugment
   megatron.data.blendable_dataset
   megatron.data.gpt_dataset
   megatron.data.image_folder
   megatron.data.realm_dataset_utils
   megatron.data.bert_dataset
   megatron.data.data_samplers
   megatron.data.indexed_dataset
   megatron.data.orqa_wiki_dataset
   megatron.data.realm_index
   megatron.data.biencoder_dataset_utils
   megatron.data.dataset_utils
   megatron.data.ict_dataset
   megatron.data.t5_dataset

megatron.model
--------------

.. autosummary::
   :toctree: megatron/model

   megatron.model.bert_model
   megatron.model.biencoder_model
   megatron.model.classification
   megatron.model.distributed
   megatron.model.enums
   megatron.model.falcon_model
   megatron.model.fused_bias_gelu
   megatron.model.fused_layer_norm
   megatron.model.fused_softmax
   megatron.model.glu_activations
   megatron.model.gpt_model
   megatron.model.language_model
   megatron.model.llama_model
   megatron.model.module
   megatron.model.multiple_choice
   megatron.model.positional_embeddings
   megatron.model.t5_model
   megatron.model.transformer
   megatron.model.utils

megatron.optimizer
------------------

.. autosummary::
   :toctree: megatron/optimizer

   megatron.optimizer.clip_grads
   megatron.optimizer.distrib_optimizer
   megatron.optimizer.grad_scaler
   megatron.optimizer.optimizer

megatron.text_generation
------------------------

.. autosummary::
   :toctree: megatron/text_generation

   megatron.text_generation.api
   megatron.text_generation.beam_utils
   megatron.text_generation.communication
   megatron.text_generation.forward_step
   megatron.text_generation.generation
   megatron.text_generation.sampling
   megatron.text_generation.tokenization

megatron.tokenizer
------------------

.. autosummary::
   :toctree: megatron/tokenizer

   megatron.tokenizer.bert_tokenization
   megatron.tokenizer.gpt2_tokenization
   megatron.tokenizer.tokenizer
