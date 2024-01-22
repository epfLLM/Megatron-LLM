import queue
from megatron import get_args


class WeightGradStore:

    cache = []
    weight_grad_queue = queue.Queue()
    split_bw = True

    @classmethod
    def is_supported(cls):
        """If not supported, fallback to original schedule."""
        args = get_args()
        if args.pipeline_model_parallel_size <= 1:
            return False
        if args.virtual_pipeline_model_parallel_size is not None:
            return False
        if args.transformer_impl == 'transformer_engine':
            # hard to capture weight gradient computation for transformer_engine
            return False
        return True

    @classmethod
    def put(cls, total_input, grad_output, weight, func):
        if not cls.split_bw or not cls.is_supported():
            func(total_input, grad_output, weight.main_grad)
            return
        # Store the weight gradient computation of linear layers.
        cls.cache.append((total_input, grad_output, weight, func))

    @classmethod
    def flush(cls):
        if not cls.is_supported():
            return
        # Collect all stored computations during backward as a W.
        cls.weight_grad_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls):
        if not cls.is_supported():
            return
        # Execute a single W.
        assert cls.weight_grad_queue.qsize() > 0
        stored_grads = cls.weight_grad_queue.get()
        for total_input, grad_output, weight, func in stored_grads:
            func(total_input, grad_output, weight.main_grad)

    @classmethod
    def pop_all(cls):
        # Execute all remaining W.
        remaining_qsize = cls.weight_grad_queue.qsize()
        for _ in range(remaining_qsize):
            cls.pop()
