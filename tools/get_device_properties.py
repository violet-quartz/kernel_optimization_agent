"""What the accelerator this run targets can actually offer.

`check_resource_usage` reports what a kernel asks for; this reports what the
hardware has. The two are only meaningful together — 49KB of shared memory per
block is fine on one device and over the limit on another, and a grid of 32
blocks saturates nothing on a 132-SM card.

The numbers come from Triton's own driver, so they describe the device as the
compiler sees it, not as torch does.
"""

import json

from anthropic import beta_tool
from anthropic.lib.tools import ToolError


@beta_tool
def get_device_properties(device: int = 0) -> str:
    """Report the hardware limits of the accelerator the kernels run on.

    Call this before choosing block sizes, `num_warps` or pipeline depth, and
    when reading `check_resource_usage` output — a shared-memory or register
    number only means something against the device's limit.

    Args:
        device: Index of the device to query. Defaults to 0, which is the
            device the benchmark uses.

    Returns:
        A JSON object of whatever Triton's driver reports for this backend. On
        CUDA that is:
          - `max_shared_mem`: shared memory per block, in bytes. A kernel whose
            `smem` approaches this fits only one block per SM, so there is
            nothing left to hide latency with.
          - `multiprocessor_count`: number of SMs. A grid smaller than this
            leaves part of the device idle whatever the kernel does.
          - `max_num_regs`: registers available to a block. Compare against
            `n_regs` × block size to see how close spilling is.
          - `warp_size`: threads per warp — 32 on CUDA. `num_warps` × this is
            the block's thread count.
          - `sm_clock_rate` / `mem_clock_rate` (kHz) and `mem_bus_width`
            (bits): peak memory bandwidth is
            `2 × mem_clock_rate × mem_bus_width / 8`, which is what a
            bandwidth-bound kernel is really competing with.
        Other backends (npu, mlu) report their own key set.

    Raises:
        ToolError: triton is not installed, or no accelerator is available.
    """
    try:
        from triton.runtime import driver
    except ImportError as exc:
        raise ToolError(
            "triton is not installed here, so device properties cannot be read."
        ) from exc

    try:
        props = driver.active.utils.get_device_properties(device)
    except Exception as exc:
        # driver.active resolves lazily and raises when no accelerator is
        # present, which is a fact about the machine, not a bug in the kernel.
        raise ToolError(
            f"could not read properties of device {device}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # The tool runner puts this value straight into the request body, where
    # tool_result content must be a string — a dict there is rejected.
    return json.dumps(props, indent=2, ensure_ascii=False, default=str)
