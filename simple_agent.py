import json

from anthropic import Anthropic, beta_tool
from dotenv import load_dotenv

from tools.bench import bench_mark, check_correctness
from tools.check_resource_usage import check_resource_usage
from tools.check_ttgir import check_ttgir
from tools.get_device_properties import get_device_properties
from tools.record import load_history, make_kernel_tools
from tools.run_dir import start_run


@beta_tool
def read_file(file_path: str) -> str:
    """Read a file, get its content.

    Args:
        file_path: file path
    Returns:
        A string with all the file content
    """
    with open(file_path, mode="r", encoding="utf-8") as file:
        return file.read()


def dump_transcript(run, runner) -> None:
    """Write the whole conversation to `transcript.jsonl`, one message per line.

    The runner keeps the conversation in `_params["messages"]` — assistant
    messages and the tool_result messages it builds itself — and exposes no
    public accessor, so this reaches for the private field and degrades to a
    warning if a future SDK moves it. Everything printed during the run goes to
    stdout only; without this, the reasoning behind a kernel is gone as soon as
    the terminal scrolls, and only `strategy` / `notes` survive.
    """
    params = getattr(runner, "_params", None)
    messages = params.get("messages") if isinstance(params, dict) else None
    if not messages:
        print("WARN could not read the conversation off the runner; "
              "no transcript was written")
        return
    try:
        with run.transcript_path.open("w", encoding="utf-8") as handle:
            for message in messages:
                payload = (
                    message.model_dump()
                    if hasattr(message, "model_dump")
                    else message
                )
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        # The run itself is already finished — a failed dump must not swallow
        # the summary that follows.
        print(f"WARN failed to write {run.transcript_path}: {exc}")


MODEL = "deepseek-v4-pro"
KERNEL_FILE_PATH = "tasks/centre_random_augmentation.py"
MAX_ATTEMPTS = 6


def main():
    load_dotenv()
    client = Anthropic()
    run = start_run(KERNEL_FILE_PATH, model=MODEL)
    write_triton_kernel, record_result = make_kernel_tools(run)
    Tools = [
        read_file,
        write_triton_kernel,
        check_correctness,
        bench_mark,
        record_result,
        # Diagnostics: optional, and only worth their tokens when a version is
        # correct but slower than it should be.
        check_resource_usage,
        check_ttgir,
        get_device_properties,
    ]

    # The rules are the same on every turn and for every task, so they live in the
    # system prompt: it keeps the cacheable prefix stable and leaves the user turn
    # carrying only what actually changes between runs.
    SYSTEM = f"""你是一个 Triton 算子优化 agent。给定一个 PyTorch 参考实现，你要写出等价但更快的实现，把加速比（speedup）做到尽可能高。

## 开工前
先 `read_file("knowledge/triton_playbook.md")`，按里面的分类判断当前算子属于哪一类、通常该用哪些手法；再 `get_device_properties()` 拿到这台机器的硬件上限。两者都读完再写第一版。手册第 3 节是「诊断信号 → 该做什么」的对照表，后面每一版遇到问题都回去查。

## 工作循环
每一轮依次做完这四步，然后开始下一轮：

1. `write_triton_kernel(code, strategy)` —— 保存新版本，返回 `version` / `kernel_path` / `v0_path`。
2. `check_correctness(v0_file=v0_path, v1_file=kernel_path)` —— 正确性没过，性能数字没有意义。
3. `bench_mark(v0_file=v0_path, v1_file=kernel_path)` —— 拿到 `v0_median_ms` / `v1_median_ms` / `speedup`。
4. `record_result(version, notes)` —— **每一版都要记录，编译失败和结果错误的版本同样要记录。** 正确性和耗时不用你填：第 2、3 步实测的结果已经落盘，这里直接读；你只写 `notes`。没跑过第 2 步就调它会被拒。

第 4 步不能跳过：它维护 best 指针和历史，是你下一轮避免重复踩坑的唯一依据。它的返回值里有 `is_best` 和当前 leaderboard。

第 3 步拿到数字之后、动手写下一版之前，先用下面的信息工具弄清楚这个数字是怎么来的。

## 信息工具
这三个不产生新版本、不改变任何东西，它们的作用是把「这份代码在这台机器上实际是什么样」摆到你面前，让下一版的改动有依据。**每写一版之前，手上都应该已经有这些信息**——只盯着 speedup，你能做的就只有随机调参数：

- `get_device_properties()` —— 硬件上限：`max_shared_mem`、`max_num_regs`、`multiprocessor_count`、`warp_size`、时钟与位宽。**开工时就调一次**：BLOCK 大小、`num_warps`、grid 划分都要对着这些数字定，凭空猜是在浪费版本预算。
- `check_resource_usage(v1_file=kernel_path)` —— 每个 kernel 的 `n_regs` / `n_spills` / `smem`。`n_spills > 0` 说明寄存器不够用、溢出到了 local memory，这是手写 Triton kernel 变慢最常见的原因，而且在计时数字上完全看不出来。
- `check_ttgir(v1_file=kernel_path)` —— 编译器实际生成了什么：`layout_conversions`（每一个都是源码没要求的共享内存往返）、`reductions`、`async_copies`（循环里为 0 说明访存没流水起来）、`dots`（matmul 形状的 kernel 为 0 说明没走上 tensor core）、以及选中的 `#blocked` / `#mma` / `#shared` 布局。

这些信息本身不会让 kernel 变快 —— 它们的价值在于把「下一版改什么」从猜测变成推断：哪一处触到了硬件上限、编译器把你的代码变成了什么、你以为发生了的事情有没有真的发生。所以后两个在任何一版跑通之后都值得调一次，不必等到结果不理想。把看到的结论写进 `record_result` 的 `notes`，下一轮才用得上。

## 提交代码的契约
`write_triton_kernel` 会在写盘前校验，不满足会被拒绝（不消耗版本号，直接改了重提）：

- 顶层必须有 `class ModelNew` 且含 `forward` 方法。不能写成 `ModelNew = Model` 这类赋值，会被加载器丢弃。
- 顶层必须同时有 `get_init_inputs` 和 `get_inputs` 两个函数。
- `ModelNew` 要能用 `*get_init_inputs()` 构造，且 `state_dict` 的 key 和 shape 与 `Model` 完全一致 —— 评测会把 v0 的权重拷进 `ModelNew`，结构不一致就无法比较。
- 模块级常量必须是字面量。`DEV = torch.device('cuda')` 这类计算出来的顶层赋值会被静默丢弃，随后在 kernel 里 `NameError`。要么放进函数或 `__init__`，要么写成字面量。
- 文件是独立加载的，用到的 import 必须写全。

## 规则
- 禁止用 try/except、if 分支等手段绕开自定义算子、回退到 PyTorch 内置实现。要么算子真的跑通，要么如实记录失败。
- 正确性没通过时，不要汇报加速比。
- 不要在第一个正确的版本上停下。在预算内尝试**多种不同思路**（tile/block 大小、访存合并、算子融合、消除中间张量、更好的 grid 划分、减少 kernel 启动次数），而不是反复微调同一版。
- `strategy` 写清楚这一版改了什么、为什么应该更快 —— 它会回灌给后续轮次。
- 正确性失败时先判断是数值精度还是索引/边界问题，改完开新版本。**不要靠放宽容差来通过。**

## 结束
预算用完后，用一段话总结：最快的是哪个版本、speedup 多少、它赢在哪里、还有哪些思路没来得及试。"""

    user_input = f"""优化目标：`{run.v0_path}`

先用 `read_file` 读这个文件，说明它在计算什么、哪一步最可能是瓶颈，再开始写第一版。

本次预算：{MAX_ATTEMPTS} 个版本。"""

    runner = client.beta.messages.tool_runner(
        model=MODEL,
        # A full Triton kernel does not fit in 1024 tokens — a truncated response
        # costs a whole attempt.
        max_tokens=16000,
        system=SYSTEM,
        tools=Tools,
        messages=[{"role": "user", "content": user_input}],
        # One attempt is ~4 tool calls plus the turns around them, so the budget
        # has to scale with MAX_ATTEMPTS or the run stops mid-search.
        max_iterations=MAX_ATTEMPTS * 6 + 10,
    )

    print(f"run: {run.root}")
    for message in runner:
        for block in message.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[assistant] {block.text.strip()}")
            elif block.type in ("thinking", "redacted_thinking"):
                # Only present when the endpoint is asked for extended thinking;
                # without it the reasoning arrives as ordinary text blocks.
                thinking = getattr(block, "thinking", "") or "<redacted>"
                print(f"\n[thinking] {thinking.strip()}")
            elif block.type == "tool_use":
                preview = {k: v for k, v in block.input.items() if k != "code"}
                if "code" in block.input:
                    preview["code"] = f"<{len(block.input['code'])} chars>"
                print(f"\n[tool] {block.name}({preview})")

    dump_transcript(run, runner)

    print(f"\n{'=' * 60}")
    print(f"run: {run.root}")
    print(f"transcript: {run.transcript_path}")
    print(f"best: {run.read_meta().get('best')}")
    for row in load_history(run):
        status = f"{row['speedup']:.2f}x" if row.get("speedup") else ("ok" if row["correct"] else "FAIL")
        print(f"  {row['version']}  {status:>7}  {row.get('strategy') or ''}")


if __name__ == "__main__":
    main()
