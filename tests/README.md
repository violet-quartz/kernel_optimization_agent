# 本地测试

只依赖标准库 `unittest`，开发机和 GPU 机器上跑法完全一样：

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t .
```

54 个用例，纯 CPU，不到 1 秒。

## 为什么值得写

到目前为止真正打断过 GPU run 的 bug，没有一个是 kernel 写错了，全是**工具契约**错了：

- `bench_mark` 返回 `dict` —— `tool_result.content` 只接受字符串或 content block 列表，请求直接 400
- `bench_mark(v0_file: str)` 注解成 `str`，随后 `.resolve()` —— `AttributeError`
- `check_input_file_path` 抛 `SystemExit`，而 runner 只 `except Exception` —— 整个 agent 循环被掀掉
- `atol: float = None` —— 模型发 `null` 时 pydantic 拒绝

这些错误都要跑到第几轮、烧掉真金白银才暴露出来，但在本地全都是毫秒级可查的。

## 覆盖范围

| 文件 | 覆盖 |
| --- | --- |
| `test_tool_contracts.py` | 每个 `@beta_tool`：schema 里每个参数都有具体类型和描述；默认值是 `None` 的参数必须能接受 `null`；返回值必须是文本；坏输入必须抛 runner 能捕获的 `Exception` |
| `test_run_dir.py` | 目录布局、v0 冻结副本、版本号分配、`best` 相对软链（run 可整体移动）、`latest` 查找 |
| `test_record.py` | `write_triton_kernel` 的 6 条校验规则逐条、被拒不消耗版本号、speedup 由 harness 计算、best 只在「正确且更快」时移动 |
| `test_auto_bench_units.py` | 加载器的字面量过滤、`record._is_safe_literal` 与 `auto_bench_v2` 那份是否仍然一致、`build_case`、路径校验 |

`tests/_helpers.py` 里的 `V1_SOURCE` 故意只依赖 torch 不 import triton —— 会真正**加载**这个文件的用例，否则在没装 triton 的开发机上跑不了。

## 没有覆盖的部分

`check_correctness` 和 `bench_mark` 按设计就需要加速器（`_detect_target_device` 拒绝在 CPU 上比较）。所以：

- 开发机上只断言它们**报错报得清楚**，而不是悄悄拿 CPU 张量计时、给出一个没有意义的加速比
- 同一批用例放到 GPU 机器上会自动走真实路径（检测到加速器就 skip 掉 CPU 断言）

计时精度本身不测，靠锁频解决 —— `bench_mark` 会对 stdev/median > 10% 的版本报 warning：

```bash
nvidia-smi -pm 1
nvidia-smi --lock-gpu-clocks=<base>,<base>
```

## 维护约定

工具的签名或返回值一改，就在这里补一条。这类改动的失败位置在几轮对话之后、在付费的 run 里面，本地一条断言就能挡住。
