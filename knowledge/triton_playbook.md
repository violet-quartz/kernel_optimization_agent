# Triton 优化手册

如何判断瓶颈并进行优化

## 1 获得硬件特性

可以通过提供的 get_device_properties 函数获取

## 2 首先判断是 memory-bound 还是 compute-bound

kernel 跑的慢，有两类瓶颈：

- memory-bound（访存受限）：计算很少，但要搬很多数据。GPU 的计算单元大部分时间在等数据从显存运过来，算力闲着。瓶颈是显存带宽
- compute-bound（计算受限）：数据不多，但要做大量运算。数据早就到位了，GPU 的计算单元在满负荷算。瓶颈是算力（FLOPS）。

如何判断是哪一类瓶颈，可以使用 Roofline 模型，计算算数强度与硬件的 ridge point（脊点） 进行比较：

**算术强度** = FLOP 数（做多少次浮点运算） / 字节数（搬多少数据）

它衡量的是“每从显存搬一字节数据，能做多少次计算”。强度低，则搬的多、算得少，是 memory-bound；
强度高，则算的多、搬的少，是 compute-bound。**ridge point（脊点）** 是硬件的一个固有分界值 = 硬件峰值算力 ÷ 峰值带宽，
算数强度 < ridge point 是 memory-bound, 算数强度 > ridge point 是 compute-bound。

| 类型 | 特征 | 优化方向 |
|---|---|---|
| **memory-bound** | 算术强度低于 ridge point。elementwise、norm、softmax、绝大多数算子 | **减少 DRAM 往返** = fusion |
| **compute-bound** | matmul、attention | **喂饱 tensor core** = tiling、流水、L2 复用 |


## 3 计算带宽百分比

```python
ms = triton.testing.do_bench(lambda: kernel_call(), return_mode='median') # 单次执行的 ms 数
gbps = bytes_moved * 1e-9 / (ms * 1e-3)
print(f"{gbps:.0f} GB/s = {gbps/REF_GBPS:.0%}")   # REF_GBPS：硬件峰值带宽
```
这里的 bytes_moved 可以是理论上的最小值：输入字节数 + 输出字节数，也可以是实际的 DRAM 访存量。

| 指标 | bytes_moved | 用途 |
|---|---|---|
| **有效吞吐** | 理论最小流量 | 同样的理论最小流量，带宽百分比高的算的更快，用于横向比较不同的实现 |
| **实际 DRAM 带宽** | 真实访存量 | 判断某实现还有没有空间 |


我们可以计算有效吞吐，然后除以峰值带宽，≥80% 就收工，50~80% 调参，<50% 往下查。


## 4 Fusion：解决 memory-bound 问题

`y = relu(x*a+b)` 在 PyTorch 里是三次 kernel 启动、6 次 DRAM 访问；融合成一个 kernel 后是 2 次。**访存降到 1/3，速度快 3 倍。**

寄存器/shared memory 比 DRAM 快一两个数量级，数据一旦读进寄存器，在里面做多少次运算几乎免费。


## 5 检查寄存器 spill 

**`n_spills > 0` 是性能灾难**——寄存器装不下，溢出到 local memory（其实在显存里）。

可以通过 check_resource_usage 函数获取。

## 6 occupancy

occupancy（占用率）的含义：一个 SM 上能同时驻留多少个 block。

```python
# 按寄存器算能放几个 block
occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
# 按 shared memory 算能放几个 block，取更小值
occupancy = min(occupancy, SIZE_SMEM // size_smem)
```

所以当 occupancy 高的时候，访存延迟容易被计算填满，使硬件利用率增加，但是 occupancy 过高的话，意味着一个 SM 上的 block 更多，
那么每个 block、每个线程能分到的寄存器更少，寄存器不够的话，出发 spill，得不偿失。occupancy 需要设一个足够藏延迟又不出发 spill 的值。


## 7 L2 复用（compute-bound 专属）

matmul 的 swizzle：把线性 pid 重排成"分组列主序"，让同时活跃的 program 覆盖一个接近**方形**的输出区域。

原理：同样数量的输出 tile，排成方块比排成长条需要的输入数据少得多（周长最小）。9×9 网格算 9 个 tile，行主序要读 90 个 block，3×3 分组只要 54 个。

## 8 参数调优：交给 autotune

`BLOCK_SIZE`、`num_warps`、`num_stages` 的最优值**推不出来**，受寄存器压力、occupancy、L2 命中率的耦合影响，还随硬件和 shape 变化。

```python
@triton.autotune(configs=[...], key=['M', 'N', 'K'])
```


## 9 访存合并

尽量让 **stride=1 的那一维是连续维，访存效率最高。** 让 block 的最内层沿着这个方向展开，相邻线程读相邻地址，硬件可以把多次请求打包成一次事务。

对于非连续的输入，要么传 stride 让 kernel 处理（省拷贝，访存可能不合并），或者 `.contiguous()` 物化（多一次拷贝，后续高效）。matmul 对 B 通常选后者，elementwise 通常选前者。

## 10 算法层面的重构


## 11 查看 TTGIR

TTGIR（这一层）：硬件相关。它在 TTIR 的基础上，加入了数据布局（layout）、线程/warp 如何分工、用不用 shared memory、tensor core 怎么调度等 GPU 专属的决策。TTGIR 可以通过 check_ttgir 函数获取。