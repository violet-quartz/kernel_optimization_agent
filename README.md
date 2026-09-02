# kernel_optimization_agent

一个**不用 agent 框架**的 Triton 算子优化 agent。

## 初衷

写这个 repo 是为了搞清楚一个 agent 到底由哪些部件组成——不借助任何 agent 框架，所有环节自己写一遍：工具怎么定义、循环怎么转、失败怎么回灌给模型、每一步的结果记在哪。

选「优化 Triton kernel」作为任务，一个好处它自带一个**客观的评分函数**：正确性能验、加速比能测，这让 agent 的每一轮都有真实反馈，而不是靠自我评价打转。

## Runner 其实就是一个 while 循环

`simple_agent.py` 用的是 SDK 里的 `client.beta.messages.tool_runner`。它**不是框架**，就是 `anthropic/lib/tools/_beta_runner.py` 里三十行的一个循环，剥掉之后是这样：

```python
messages = [{"role": "user", "content": user_input}]
by_name = {t.name: t for t in Tools}

for _ in range(MAX_ITERATIONS):
    msg = client.beta.messages.create(model=MODEL, max_tokens=16000,
                                      system=SYSTEM, tools=Tools, messages=messages)
    messages.append({"role": "assistant", "content": msg.content})   # 必须原样回传

    tool_uses = [b for b in msg.content if b.type == "tool_use"]
    if not tool_uses:
        break                                                        # 没有工具调用 = 该结束了

    results = []
    for tu in tool_uses:
        try:
            content = by_name[tu.name].call(tu.input)
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": content})
        except Exception as e:                                       # 关键：回给模型，别往外抛
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": f"Error: {e}", "is_error": True})
    messages.append({"role": "user", "content": results})
```

理解 agent 原理，理解这段就够了。两点需要注意的：

1. **工具异常要转成 `is_error=True` 的 `tool_result`，不能往外抛。** 这是 agent 能自我修复的唯一机制——`write_triton_kernel` 抛出「ModelNew 必须是 class」，模型下一轮才能改对。让异常穿透，整个 agent 就挂了。
2. **一轮里可能有多个 `tool_use`**，它们的结果要装进**同一条** user 消息。

`tool_runner` 比这多出来的只有边角：`refusal` 停止、未知工具的告警、响应缓存、中途增删工具的钩子。

## 目录

```
simple_agent.py     优化 agent 主程序：工具清单 + system prompt + 主循环
mini_agent.py       最小示例：两个工具（读文件、跑 Python），用来看清最小闭环
knowledge/          喂给模型的静态知识，模型开工前自己 read_file 读
tasks/              待优化的 PyTorch 参考实现（v0）
tools/              所有工具
  bench.py                 check_correctness / bench_mark —— 正确性与加速比
  record.py                write_triton_kernel / record_result —— 版本与历史
  run_dir.py               一次 run 的目录布局与版本分配
  check_resource_usage.py  每个 kernel 的 n_regs / n_spills / smem
  check_ttgir.py           编译器实际生成的 TTGIR：layout 转换、流水、tensor core
  get_device_properties.py 硬件上限
runs/               每次运行的产物
tests/              单元测试
```

工具分两类：**改变状态的**（写版本、记结果）和**只提供信息的**（三个诊断工具）。后者不产生新版本，作用是让模型的下一次修改有依据，而不是靠猜着调参数。

## 跑起来

```bash
pip install -r requirements.txt          # torch 见 requirements.txt 里的说明，按卡装
echo 'ANTHROPIC_API_KEY=...' > .env       # 另可设 ANTHROPIC_BASE_URL
python simple_agent.py                    # MODEL / KERNEL_FILE_PATH / MAX_ATTEMPTS 在文件顶部
```

需要一块加速器（cuda / npu / mlu）——正确性和性能都在真实设备上测，CPU 上跑不了。

## 一次 run 留下什么

```
runs/<时间戳>_<任务名>/
  v0.py             参考实现的快照
  meta.json         模型、任务、环境信息、当前 best
  history.jsonl     每一版一行，回灌给模型的记忆
  transcript.jsonl  完整对话：推理、工具入参、工具返回值
  best -> v00N      指向当前最快且正确的版本
  v001/
    kernel.py       模型写的代码
    attempt.json    版本号、父版本、strategy（写之前就落盘）
    bench.json      check_correctness / bench_mark 的实测结果
    result.json     最终记录：correct / 耗时 / speedup / notes
```

两条设计原则：

- **数据由 harness 记录，不由模型自述。** `record_result` 只接受 `version` 和 `notes`；正确性和耗时是从 `bench.json` 读的，模型无法覆盖。否则 leaderboard 建立在模型的自我汇报上，一次抄错数字就全歪了。
- **工具的 docstring 就是给模型的说明书。** 它会被 `@beta_tool` 转成 schema 连同描述一起发给模型，所以每个返回字段该怎么解读都写在 docstring 里——只给一串数字，模型判断不了。

