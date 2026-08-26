from anthropic import Anthropic, beta_tool
from dotenv import load_dotenv

from tools.auto_bench_v2 import bench_mark, check_correctness
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


load_dotenv()
client = Anthropic()
MODEL = "deepseek-v4-pro"
KERNEL_FILE_PATH = "tasks/centre_random_augmentation.py"
MAX_ATTEMPTS = 6

run = start_run(KERNEL_FILE_PATH, model=MODEL)
write_triton_kernel, record_result = make_kernel_tools(run)
Tools = [read_file, write_triton_kernel, check_correctness, bench_mark, record_result]

# The rules are the same on every turn and for every task, so they live in the
# system prompt: it keeps the cacheable prefix stable and leaves the user turn
# carrying only what actually changes between runs.
SYSTEM = f"""你是一个 Triton 算子优化 agent。给定一个 PyTorch 参考实现，你要写出等价但更快的实现，把加速比（speedup）做到尽可能高。

## 工作循环
每一轮依次做完这四步，然后开始下一轮：

1. `write_triton_kernel(code, strategy)` —— 保存新版本，返回 `version` / `kernel_path` / `v0_path`。
2. `check_correctness(v0_file=v0_path, v1_file=kernel_path)` —— 正确性没过，性能数字没有意义。
3. `bench_mark(v0_file=v0_path, v1_file=kernel_path)` —— 拿到 `v0_median_ms` / `v1_median_ms` / `speedup`。
4. `record_result(version, correct, v0_ms, v1_ms, error, notes)` —— **每一版都要记录，编译失败和结果错误的版本同样要记录。**

第 4 步不能跳过：它维护 best 指针和历史，是你下一轮避免重复踩坑的唯一依据。它的返回值里有 `is_best` 和当前 leaderboard。

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
        elif block.type == "tool_use":
            preview = {k: v for k, v in block.input.items() if k != "code"}
            if "code" in block.input:
                preview["code"] = f"<{len(block.input['code'])} chars>"
            print(f"\n[tool] {block.name}({preview})")

print(f"\n{'=' * 60}")
print(f"run: {run.root}")
print(f"best: {run.read_meta().get('best')}")
for row in load_history(run):
    status = f"{row['speedup']:.2f}x" if row.get("speedup") else ("ok" if row["correct"] else "FAIL")
    print(f"  {row['version']}  {status:>7}  {row.get('strategy') or ''}")
