# 技术细节：自迭代回路设计

本文档详解自迭代抓取系统的五个核心模块，以及实验中的关键决策。

## 1. 结构化失败反馈

原系统失败时只返回字符串（"物体未被真实夹起"），无法供 LLM 推理。我们升级为结构化字典：

```python
feedback = {
    "success": False,
    "final_state": "LIFT",          # 卡在哪个 FSM 状态
    "reached_grasp_pos": True,      # 末端是否到达抓取点
    "lift_delta": 0.0076,           # 真实抬升（核心物理信号）
    "lift_threshold": 0.06,         # 成功阈值
    "obj_xy_drift": 0.0053,         # 下降时物体被撞飞的距离
    "params_used": {...}            # 本次使用的参数
}
```

**关键点**：`lift_delta` 在夹持辅助介入前采集，是纯净的物理信号。LLM 读到 `lift=0.008 < 0.06 且 reached=True` 即可推理出"抓空了，应该抓更深"。

## 2. 参数可注入

抓取参数从写死的全局常量改为可注入字典，使 LLM 的调整能真正生效：

```python
def execute_grasp_fsm(env, obs, target_name, ..., params=None):
    params = clip_grasp_params(params) if params else default_grasp_params(target_name)
    grasp_z_offset = params["z_offset"]
    close_steps = int(params["close_steps"])
    descend_gain = float(params["descend_gain"])
```

`params=None` 时回退到默认值，保证不开自迭代时行为完全不变（向后兼容）。

## 3. LLM 调参 + 启发式回退

```python
def llm_refine_grasp_params(target_name, history):
    # 启发式回退：LLM 不可用时仍能自迭代
    def heuristic_tweak():
        if lift < threshold * 0.3:
            tweaked["z_offset"] -= 0.025      # 抓更深
            tweaked["close_steps"] += 20      # 夹更久
        if drift > 0.04:
            tweaked["descend_gain"] -= 0.8    # 下降更柔
        return clip_grasp_params(tweaked)

    if not api_key:
        return heuristic_tweak()
    # ... 调 Qwen-VL，失败则回退 heuristic_tweak()
```

**双保险**：LLM 失败 → 启发式保底；输出强制 clip → 防止越界。

## 4. 形状专属初始参数（V4 关键改进）

V4 的 75% 突破来自给不同几何形状的物体设专属起点：

```python
if target_name in {"bread"}:      # 平面：抓深 + 夹久
    params.update(z_offset=0.005, close_steps=85, descend_gain=3.2)
elif target_name in {"cereal"}:   # 薄盒：更深 + 更久 + 更柔
    params.update(z_offset=0.0, close_steps=95, descend_gain=2.8)
elif target_name in {"milk"}:     # 高圆柱：中段抓
    params.update(z_offset=0.03, close_steps=90, descend_gain=3.0)
elif target_name in {"can"}:      # 矮圆柱：抓深
    params.update(z_offset=0.0, close_steps=85, descend_gain=3.0)
```

效果：bread/milk/can 都在第 2 轮成功（V1-V3 需要 3-4 轮）。

## 5. CLI 开关 + Ablation 日志

```python
parser.add_argument("--self-iterate", action="store_true")  # 默认关，不破坏原 demo
```

每次尝试写一行 JSONL，作为 ablation 数据：

```json
{"target": "bread", "attempt": 1, "params": {...}, "success": false, "lift_delta": 0.003, ...}
```

## 四版演进总结

| 版本 | 核心思路 | 解决率 | 学到的 |
|---|---|---|---|
| V1 | 基础自迭代 | 25% | 回路能跑通 |
| V2 | LLM 调参 | 0% | LLM 需要更强约束 |
| V3 | 加轮次 + 降阈值 + 分级启发式 | 50% | 理解物理约束 |
| V4 | 形状专属初始参数 | 75% | 先验加速收敛 |

## 失败案例分析：cereal

cereal（薄盒）8 轮全失败，`reached_grasp_pos` 多数为 False：
- **根因**：薄盒在 MuJoCo 里碰撞面小，末端稍偏就碰不到
- **现有方案不足**：随机 xy 探索方向无引导，命中率低
- **下一步**：系统性 xy 网格搜索 / 侧面抓取 / 推送到墙角再抓
