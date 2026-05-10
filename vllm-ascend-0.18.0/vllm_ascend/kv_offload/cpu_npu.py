import numpy as np
import torch
from vllm.logger import logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionBackend  # type: ignore
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler, TransferResult, TransferSpec


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    """
    Convert a list of block IDs to a list of matching block ids,
    assuming each block is composed of actual block_size_factor blocks.
    Outputs to output tensor.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if block_ids = [0, 1, 3] and block_size_factor =  4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


class CpuNpuOffloadingHandler(OffloadingHandler):
    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        assert cpu_block_size % gpu_block_size == 0
        self.block_size_factor = cpu_block_size // gpu_block_size

        # npu streams for npu->cpu and cpu->npu.
        # K and V use independent streams for parallel DMA transfers.
        self.d2h_stream_k = torch.npu.Stream()
        self.d2h_stream_v = torch.npu.Stream()
        self.h2d_stream_k = torch.npu.Stream()
        self.h2d_stream_v = torch.npu.Stream()

        # job_id -> (k_event, v_event) transfer npu events
        self.transfer_events: dict[int, tuple[torch.npu.Event, torch.npu.Event | None]] = {}
        # list of npu events available for reuse
        self.events_pool: list[torch.npu.Event] = []

        pin_memory = is_pin_memory_available()

        # allocate cpu tensors
        logger.info("Allocating %d CPU tensors...", len(gpu_caches))
        self.npu_tensors: list[torch.Tensor] = []
        self.cpu_tensors: list[torch.Tensor] = []
        for layer_name, gpu_tensor in gpu_caches.items():
            self.npu_tensors.append(gpu_tensor)

            gpu_shape = gpu_tensor[0].shape

            num_blocks_idx = 0
            cpu_shape = list(gpu_shape)
            cpu_shape[num_blocks_idx] = num_cpu_blocks * self.block_size_factor

            logger.debug("Allocating CPU tensor of shape %r", cpu_shape)
            self.cpu_tensors.append(
                (
                    torch.zeros(
                        cpu_shape,
                        dtype=gpu_tensor[0].dtype,
                        device="cpu",
                        pin_memory=pin_memory,
                    ),
                    torch.zeros(
                        cpu_shape,
                        dtype=gpu_tensor[0].dtype,
                        device="cpu",
                        pin_memory=pin_memory,
                    ),
                )
            )

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        logger.info("start transfer_async...")
        src_spec, dst_spec = spec
        if isinstance(src_spec, CPULoadStoreSpec):
            assert isinstance(dst_spec, GPULoadStoreSpec)
            stream_k = self.h2d_stream_k
            stream_v = self.h2d_stream_v
            src_tensors = self.cpu_tensors
            dst_tensors = self.npu_tensors
            src_block_size_factor = self.block_size_factor
            dst_block_size_factor = 1
        else:
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, CPULoadStoreSpec)
            stream_k = self.d2h_stream_k
            stream_v = self.d2h_stream_v
            src_tensors = self.npu_tensors
            dst_tensors = self.cpu_tensors
            src_block_size_factor = 1
            dst_block_size_factor = self.block_size_factor

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        dst_sub_blocks_to_skip = -src_blocks.size % dst_block_size_factor
        src_sub_block_count = src_blocks.size * src_block_size_factor

        assert src_sub_block_count == dst_blocks.size * dst_block_size_factor - dst_sub_blocks_to_skip

        src_to_dst = np.empty((src_sub_block_count, 2), dtype=np.int64)
        expand_block_ids(src_blocks, src_block_size_factor, src_to_dst[:, 0])
        expand_block_ids(
            dst_blocks,
            dst_block_size_factor,
            src_to_dst[:, 1],
            skip_count=dst_sub_blocks_to_skip,
        )
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        # Allocate events for K and V parallel streams.
        event_k = self.events_pool.pop() if self.events_pool else torch.npu.Event()
        event_v = self.events_pool.pop() if self.events_pool else torch.npu.Event()

        # K transfers on stream_k
        with torch.npu.stream(stream_k):
            for src_tensor, dst_tensor in zip(src_tensors, dst_tensors):
                torch.ops._C_ascend.swap_blocks(
                    src_tensor[0], dst_tensor[0], src_to_dst_tensor)
            event_k.record(stream_k)

        # V transfers on stream_v (parallel with K)
        with torch.npu.stream(stream_v):
            for src_tensor, dst_tensor in zip(src_tensors, dst_tensors):
                torch.ops._C_ascend.swap_blocks(
                    src_tensor[1], dst_tensor[1], src_to_dst_tensor)
            event_v.record(stream_v)

        self.transfer_events[job_id] = (event_k, event_v)

        # success
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        finished_job_ids = []
        for job_id, (event_k, event_v) in self.transfer_events.items():
            if event_k.query() and (event_v is None or event_v.query()):
                results.append(
                    TransferResult(
                        job_id=job_id,
                        success=True,
                        transfer_size=None,
                        transfer_time=None,
                        transfer_type=None,
                    )
                )
                finished_job_ids.append(job_id)
                self.events_pool.append(event_k)
                if event_v is not None:
                    self.events_pool.append(event_v)
        for job_id in finished_job_ids:
            del self.transfer_events[job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        """
        Wait (block) until all specified transfer jobs are completed.
        """
        for job_id in job_ids:
            entry = self.transfer_events.get(job_id)
            if entry is not None:
                event_k, event_v = entry
                event_k.synchronize()
                if event_v is not None:
                    event_v.synchronize()
