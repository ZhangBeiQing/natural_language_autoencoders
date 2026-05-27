# NLA 优化实验手册

> 基于 Natural Language Autoencoders 项目的改进方案与消融实验设计

---

## 目录

1. [项目架构详解](#1-项目架构详解)
2. [问题分析与思考过程](#2-问题分析与思考过程)
3. [优化方案总览](#3-优化方案总览)
4. [消融实验设计](#4-消融实验设计)
5. [评估指标体系](#5-评估指标体系)
6. [代码改动清单](#6-代码改动清单)
7. [风险与应对](#7-风险与应对)

---

## 1. 项目架构详解

### 1.1 整体思路

NLA（Natural Language Autoencoders）的核心思想是：**用自然语言作为中间表示，在 LLM 的激活向量空间和人类可读文本之间建立双向映射**。

```
激活向量 ──[AV (Actor)]──> 解释文本 ──[AR (Critic)]──> 重建激活向量
```

- **AV（Activation Viewer / Actor）**：读入激活向量，输出自然语言解释
- **AR（Activation Reconstructor / Critic）**：读入自然语言解释，输出激活向量

训练目标：解释文本必须是激活向量的"充分统计量"——即 AR 能从文本完美重建原始激活。

### 1.2 数据生成管线（4 阶段）

```
Stage 0: 激活提取
  语料 → base model forward → hook 提取第 K 层 hidden states → base.parquet
  输出列: n_raw_tokens, detokenized_text_truncated, activation_vector, activation_layer, doc_id

Stage 1: 数据分割
  base.parquet → 按 doc_id 文档级分割为三个集合：
    - av_sft (25%): Actor SFT 训练集
    - ar_sft (25%): Critic SFT 训练集
    - rl (50%): RL 训练集

Stage 2: API 解释生成
  av_sft/ar_sft 子集 → 调用外部 LLM（默认 Claude Sonnet）
  → 为每行 detokenized_text_truncated 生成自然语言解释
  → 附加 api_explanation 列

Stage 3: 构建训练 Parquet
  输入 → 构建最终训练格式 + sidecar YAML：
    - av_sft.parquet: prompt(messages) + response + activation_vector
    - ar_sft.parquet: prompt(string) + activation_vector
    - rl.parquet: prompt(messages) + activation_vector（无 response）
```

**关键细节**：Stage 2 的外部 LLM 在生成解释时看到了 `detokenized_text_truncated`（原始文本），但最终 AV 的 prompt 模板中**不包含原始文本**。

### 1.3 激活向量注入机制

AV 如何"看到"激活向量？通过 **marker token embedding 替换**：

```
1. 选择 CJK 圈起来的 ideograph（如 ㊗ U+3297）作为注入标记
2. AV 的 prompt 模板中: <concept>㊗</concept>
3. 训练时，forward hook 扫描 input_ids 找到 ㊗ 的位置
4. 将该位置的 token embedding 替换为归一化后的激活向量
5. 模型在 forward 时，那个位置"看到"的就是激活向量而非原始 token embedding
```

注入函数核心逻辑（`nla/injection.py`）：

```python
def inject_at_marked_positions(input_ids, embeddings, vectors, inj_id, left_id, right_id):
    # 找到 inj_id 位置，验证左右邻居 token ID 匹配（防误报）
    # 用归一化后的 activation vector 替换该位置的 embedding
    out[b, p] = vectors[vec_idx]
```

归一化参数 `injection_scale`：
- `raw`：保持原始 L2 范数
- `sqrt_d_model`：归一化到 √d（匹配 token embedding 尺度）
- 具体数值：Qwen7B 用 150，Gemma-12B 用 80000

### 1.4 AV（Actor）的输入格式

**SFT 阶段的 AV Prompt 模板**（`_DEFAULT_ACTOR_TEMPLATE`）：

```
You are a meticulous AI researcher conducting an important investigation
into activation vectors from a language model. Your overall task is to
describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context.
You must then produce an explanation for the vector, enclosed within
<explanation> tags. The explanation consists of 2-3 text snippets
describing that vector.

Here is the vector:

<concept>㊗</concept>

Please provide an explanation.
```

**关键观察**：AV 在 RL 阶段**只能看到激活向量**（通过 marker token 注入），**看不到原始输入文本**。这是一个重大信息缺失。

### 1.5 AR（Critic）的输入格式

**Critic Prompt 模板**（`_DEFAULT_CRITIC_TEMPLATE`）：

```
Summary of the following text: <text>{explanation}</text> <summary>
```

AR 只接收解释文本，输出在最后一个 token 位置提取。Critic 模型是截断的 Transformer（只保留前 K+1 层）+ 线性 value head，输出维度为 d_model。

### 1.6 训练管线

#### 阶段一：AV SFT（Actor SFT）

```
目标: 教 AV 学会从注入的激活向量生成解释文本
方法: 标准自回归 SFT loss
数据: av_sft.parquet（有 response 列，由 Claude Sonnet 生成）
关键: 激活向量通过 forward hook 注入 marker token 位置
```

#### 阶段二：AR SFT（Critic SFT）

```
目标: 教 AR 学会从解释文本重建激活向量
方法: MSE loss (normalized)
数据: ar_sft.parquet
关键: 在最后一个 token 位置提取 value head 输出，与 gold activation 计算方向 MSE
```

#### 阶段三：RL 联合训练

```
同时训练 AV (GRPO) 和 AR (监督 MSE)

每个 rollout step:
1. NLADataSource 提供 RL parquet 的 samples
2. nla_generate.generate():
   a. 构建 input_embeds（注入 scaled activation）
   b. 发送到 SGLang 生成 explanation
   c. 提取 <explanation>...</explanation> 内容
   d. 构建 critic_tokens
3. nla_rm (reward):
   a. 通过 Ray remote 调用 critic trainer 的 critic_fwd()
   b. reward = -MSE(normalize(pred), normalize(gold))
4. Actor training (GRPO):
   - n_samples_per_prompt = 4（或 8）
   - 组内 reward 归一化计算 advantage
5. Critic training (supervised MSE):
   - 与 actor 并行训练在独立 GPU 池上
```

**Reward 计算**（`nla/reward.py`）：

```python
reward = -MSE(critic_fwd(explanation), gold_activation)
# L2 归一化后: MSE = 2(1 - cos_sim)
# cos_sim=1.0 → reward=0.0 (完美)
# cos_sim=0.0 → reward=-2.0 (正交)
# 提取失败: reward = -2.0 (等价于正交惩罚)
```

### 1.7 评估指标

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| FVE (nrm) | `1 - MSE / baseline_rawvar` | 解释了多少方差，>0 愔味着比常数预测好 |
| FVE (nrm_meannorm) | `1 - MSE / meannorm_baseline` | critic 最佳常数预测的 FVE |
| Direction MSE | L2归一化后的 MSE = `2(1-cos_sim)` | 0=完美，2=正交 |
| Cosine Similarity | `cos(pred, gold)` | 直观可读性：1.0=完美，0.9=好 |
| pred_norm / gold_norm | 向量 L2 范数 | 监控数值漂移 |

Qwen7B layer-20 参考值：
- `fve_nrm_meannorm ≈ 0.94`（归一化均值基线）
- `fve_nrm ≈ 0.72`（原始方差基线）

---

## 2. 问题分析与思考过程

### 2.1 问题的起源：NLA vs CoT vs SAE

我们首先理解了三种 LLM 内部状态解释方法的区别：

- **CoT（Chain of Thought）**：模型自愿输出的推理过程。但模型可以**说谎或隐瞒**——它可以选择不表达某些内部状态。论文发现：16 个评估检测场景中，NLA 检测到 10 个有"评估意识"，但只有 1 个被口头表达（Unverbalized Evaluation Awareness）。

- **SAE（Sparse Autoencoders）**：将激活向量分解为稀疏的原子特征。但 SAE 的特征是**不可读的向量方向**，不是人类可读的文本。需要额外的人类标注才能理解每个特征的含义。

- **NLA**：直接读取激活向量并生成人类可读的文本解释。模型**无法控制自己的激活向量**，所以 NLA 能检测到模型不会说出的信息。

### 2.2 发现的两个核心问题

#### 问题 1：Confabulation（幻觉编造）

**现象**：AV 生成的解释中包含**激活向量中并未编码的信息**。

**原因分析**：AV 本质上是一个语言模型，它会基于语言模型的先验"补全"文本，即使某些内容并不来自激活向量。激活向量只编码了有限的信息（比如"这里有一个语法结构"），但 AV 可能会编造出"这里涉及条件从句的嵌套"之类的具体细节。

**类比**：就像目击者在回忆事件时会"脑补"细节，AV 也会"脑补"解释。

#### 问题 2：Long-distance Forgetting（长距离遗忘）

**现象**：AV 的解释缺少远距离上下文信息。

**原因分析**：NLA 只注入**单层、单位置**的激活向量。但 LLM 的信息是分布式的：
- 当前位置的激活向量主要编码当前位置附近的信息
- 远距离的上下文（比如 500 个 token 之前的主题）在当前位置的残差流中衰减严重
- 但这些远距离信息对理解"模型在这里在做什么"可能至关重要

**证据**：Stage 2 的 API 解释（由 Claude Sonnet 生成）能看到完整文本，所以能捕捉远距离上下文。但 RL 阶段的 AV 只看单个激活向量，必然丢失这些信息。

### 2.3 对训练范式的质疑

我们质疑了当前 SFT + RL 的"左脚踩右脚螺旋上天"训练范式：

**质疑 1**：两阶段训练是否最优？SFT 预训练 → RL 优化，这个流程是否有更直接的替代方案？

**质疑 2**：encoder-decoder 架构是否必要？如果目标只是把激活映射成人类可读文本，是否一定要重建回去？

基于此，我们探索了替代方案：

| 方案 | 核心思路 | 优势 | 劣势 |
|------|---------|------|------|
| Gumbel-Softmax 连续松弛 | 去掉 RL，AV 输出 logits → Gumbel-Softmax → soft embeddings → AR → MSE | 端到端可微，无 RL 不稳定性 | 温度退火需要调参，soft tokens 和 hard tokens 有 gap |
| 对比学习 | 去掉 AR 和 RL，用激活相似度结构训练 | 不需要 ground truth 文本 | 无法保证文本的可读性 |
| 多任务属性预测 | 去掉 AR 和 RL，用自动可计算的属性任务 | 有明确监督信号 | 需要预定义属性，覆盖面有限 |

**结论**：Gumbel-Softmax 是最有前景的替代方案，但需要较大架构改动，短期内不作为优先方向。

### 2.4 核心洞察：最大化利用已知信息

我们转向一个更务实的思路：**不是换算法，而是最大化利用一切已知信息**。

当前 NLA 的信息流：

```
AV 输入: 仅激活向量
AR 输入: 仅解释文本
训练信号: 仅 MSE 重建误差
```

但我们实际上拥有大量已知信息被浪费了：

| 已知信息 | SFT 阶段 | RL 阶段 | 利用程度 |
|---------|---------|---------|---------|
| 激活向量 | ✅ | ✅ | 充分 |
| 输入文本（prompt） | ✅（Stage 2 API 看到了） | ❌ | **不充分** |
| 模型的实际输出（response） | ✅（用来生成 SFT 数据） | ❌ | 不充分 |
| 注意力权重/模式 | ❌ | ❌ | 完全未用 |
| 相邻层/位置的激活 | ❌ | ❌ | 完全未用 |
| 下游行为（模型接下来生成什么 token） | ❌ | ❌ | 完全未用 |
| 梯度/归因信息 | ❌ | ❌ | 完全未用 |
| 层号、位置号等结构信息 | ❌ | ❌ | 完全未用 |

**核心论点**：NLA 本质上是一个信息不足的逆问题（从单一信号推断高维语义）。解决方案不是更好的算法，而是**注入更多约束/信息**，让逆问题变得 well-posed。

### 2.5 AR 的设计原则

在讨论如何增强 AV 时，我们意识到 AR 的设计需要遵循一个关键原则：

**AR 应该只接收解释文本，不接收额外语义信息。**

原因：解释文本必须是激活向量的"充分统计量"。如果 AR 也能看到输入文本等额外信息，那解释文本就可以偷懒——反正 AR 有其他来源能推断出激活。这会破坏自编码器的信息瓶颈约束。

**例外**：结构性信息（如层号 embedding）可以给 AR，因为它不包含语义信息，不会让文本偷懒。

---

## 3. 优化方案总览

### 3.1 输入侧优化（给 AV 更多信息）

#### I1：输入文本 → AV

**现状问题**：SFT 阶段，Claude Sonnet 生成解释时看到了完整的原始文本（`detokenized_text_truncated`）。但 RL 阶段，AV 的 prompt 中**完全没有原始文本**。这是一个巨大的信息 gap。

**改进**：在 RL 阶段也将输入文本（或其摘要）作为 AV 的输入条件。

```
修改后的 AV Prompt:
"...Your overall task is to describe the semantic content of that activation vector.

The original input text is: {detokenized_text_truncated}

Here is the vector:
<concept>㊗</concept>
Please provide an explanation."
```

**预期收益**：★★★（最高——直接解决长距离遗忘问题）
**风险**：AV 可能直接从输入文本猜解释，不看激活向量 → 需要消融防作弊约束（L1）

#### I2：层号 → AV

不同层的激活向量承载不同类型的信息（低层→语法，中层→语义，高层→行为）。AV 知道自己在解读第几层，能更有针对性地生成解释。

```
修改后的 AV Prompt:
"This activation is from layer {layer_index} of a {total_layers}-layer model."
```

**预期收益**：★★（辅助性信号）
**风险**：低

#### I3：模型输出 top-k token 分布 → AV

激活向量导致了模型的输出行为。输出行为就是这个激活向量的"行为证据"。

```
修改后的 AV Prompt:
"The model's most likely next tokens: {top5_tokens_with_probs}"
```

**预期收益**：★★★（强信号——直接告诉你激活"做了什么"）
**风险**：中等（需要修改数据管线，增加存储和计算）

### 3.2 Loss 侧优化（给训练更多约束）

#### L1：消融防作弊约束

**目的**：防止 AV 学会"抄输入文本"而忽略激活向量。

```python
# 训练时额外采样一个对照样本
explain_real = AV(activation=real_activation, input_text=text)
explain_zero = AV(activation=zeros_vector, input_text=text)

# 如果两者太相似 → AV 没看激活，在抄文本
cheat_penalty = max(0, cos_sim(explain_real, explain_zero) - margin)

# 从 reward 中扣减
reward = -MSE - λ_cheat * cheat_penalty
```

**预期收益**：★★★（必须加——I1 的前提条件）
**风险**：计算成本翻倍（需要额外一次 AV 前向传播）

#### L2：行为一致性 loss

**目的**：好的解释应该能预测模型的输出行为。

```python
# 简化版：如果解释是好的，AR 重建的激活经过下游层应产生正确输出
# 实际上已被 MSE 隐含——MSE 好→重建激活接近→下游行为接近
# 额外价值在于 MSE 不够好时的补充约束
L_behavior = KL(predicted_output_dist, actual_output_dist)
```

**预期收益**：★★（部分与 MSE 重叠）
**风险**：需要额外的预测 head 或 forward pass

#### L3：注意力一致性 loss

**目的**：解释文本的关键词应该和注意力聚焦的输入位置一致。

```python
# 注意力模式告诉 AV "这个位置在看哪里"
attn_weights = model.layers[l].attn.weight  # [heads, seq, seq]
attn_focus = attn_weights[:, position, :].mean(0)  # [seq]
top_attn_tokens = [input_tokens[i] for i in attn_focus.topk(5).indices]

# 解释文本应与注意力焦点语义一致
L_attn = -cos_sim(embed(top_attn_tokens).mean(0), embed(explanation).mean(0)) + 1
```

**预期收益**：★★★（独立信息量高——注意力模式编码了"模型在关注什么"）
**风险**：需要修改数据管线（Stage 0 存 attention weights），存储和计算成本增加

#### L4：输入归因一致性 loss

**目的**：梯度归因告诉我们哪些输入 token 对当前激活贡献最大，解释应与之一致。

```python
attribution = ∂activation / ∂input_embedding
top_attrib_tokens = [input_tokens[i] for i in attribution.norm(dim=-1).topk(5).indices]
L_attribution = -cos_sim(embed(top_attrib_tokens).mean(0), embed(explanation).mean(0)) + 1
```

**预期收益**：★★（与 L3 有信息重叠）
**风险**：每个训练 step 需要额外 backward，计算成本最高

### 3.3 Loss 优先级评估

| Loss | 独立信息量 | 与 MSE 重叠 | 实现成本 | 建议 |
|------|-----------|-----------|---------|------|
| L1 消融防作弊 | ★★★ | 无 | ⭐⭐ | **必须加** |
| L2 行为一致性 | ★★ | 部分重叠 | ⭐⭐⭐ | 可选 |
| L3 注意力一致性 | ★★★ | 无 | ⭐⭐⭐⭐ | 有价值但成本高 |
| L4 输入归因 | ★★ | 与 L3 重叠 | ⭐⭐⭐⭐⭐ | 性价比最低，暂不做 |

---

## 4. 消融实验设计

### 4.1 实验原则

1. **从简到繁**：先加输入侧信息（几乎不影响训练稳定性），再加 Loss 侧约束（涉及多 loss 平衡）
2. **渐进式**：每步只加一个变量，明确归因
3. **每阶段取最优**：某步效果变差则调试或放弃
4. **AR 不变**：所有实验中 AR 只接收解释文本（可选加层号 embedding），保持信息瓶颈在文本上

### 4.2 阶段一：输入侧消融（低成本高收益）

| 实验 | AV 额外输入 | AR 输入 | Loss | 对比基线 | 主要验证 |
|------|-----------|---------|------|---------|---------|
| E0 | 无（当前 baseline） | 解释文本 | MSE | — | 建立基线数值 |
| E1 | +I1（输入文本） | 解释文本 | MSE | E0 | 输入文本是否帮助 AV |
| E2 | +I1+I2（输入文本+层号） | 解释文本 | MSE | E1 | 层号是否额外帮助 |
| E3 | +I1+I2+I3（+输出分布） | 解释文本 | MSE | E2 | 输出分布是否额外帮助 |

#### E0：Baseline 复现

**目标**：复现作者的指标，建立数值基线。

**操作**：
1. 使用项目现有代码和默认配置训练
2. 记录：FVE (nrm)、FVE (nrm_meannorm)、Direction MSE、Cosine Similarity
3. 记录训练曲线：这些指标随 training step 的变化

**验收标准**：FVE (nrm) 应 ≥ 0.72 的某个比例（因为 FVE 是 predict-mean baseline，模型应超过它）

#### E1：给 AV 加输入文本

**目标**：验证输入文本是否帮助 AV 生成更好的解释。

**关键代码改动**：

1. **数据侧**：确认 RL parquet 中是否存有 `detokenized_text_truncated` 列
   - 查看 `nla/datagen/stage3_build.py` 中 `_schema_for("rl")` 和 `_PROVENANCE_COLS`
   - 当前 `_PROVENANCE_COLS = ["n_raw_tokens", "activation_layer", "doc_id"]`，**不包含** `detokenized_text_truncated`
   - 如果有 `--keep-debug-metadata`，则 `detokenized_text_truncated` 会被带入
   - **如果 RL parquet 没有该列**：需要修改 `_PROVENANCE_COLS` 或在 stage3_build 中强制携带

2. **AV Prompt 模板**：修改 `_DEFAULT_ACTOR_TEMPLATE`，追加输入文本段落

3. **NLADataSource**：确认 `detokenized_text_truncated` 被正确加载到 sample.metadata

4. **nla_generate.py**：在构建 input_embeds 时，将输入文本编入 prompt

**消融检查**：
- 比较有/无输入文本时 AV 生成的解释差异
- 检查 AV 是否过度依赖输入文本（通过 E4 的消融约束量化）

**风险**：
- 如果输入文本很长，会占用大量 context window，可能影响解释生成的 token 预算
- 缓解：只取截断文本的最后 N 个 token（比如最后 200 个）

#### E2：给 AV 加层号

**操作**：在 AV prompt 中追加一句话说明当前层号。

**消融检查**：比较 E2 vs E1 的 FVE 变化。

**风险**：低。即使层号没有帮助，也不太可能有害。

#### E3：给 AV 加输出 token 分布

**关键代码改动**：

1. **Stage 0 数据提取**：在 `nla/datagen/stage0_extract.py` 中，提取激活向量时同时记录该位置的 top-k 输出 token 及其概率
2. **Parquet schema**：增加 `top_output_tokens` 列
3. **AV Prompt 模板**：追加输出 token 信息

**风险**：
- 数据管线改动较大
- top-k token 可能泄露过多信息，让 AV 学会"抄答案"而非真正理解激活

### 4.3 阶段二：Loss 侧消融（中等成本）

**前提**：阶段一完成后，选取最优的输入侧组合。

| 实验 | 输入（取 P1 最优） | Loss 改动 | 对比基线 | 主要验证 |
|------|-------------------|----------|---------|---------|
| E4 | P1 最优 | +L1（消融防作弊） | P1 最优 | AV 是否真的在看激活向量 |
| E5 | P1 最优 | +L2（行为一致性） | P1 最优 | 行为约束是否有额外信息 |
| E6 | P1 最优 | +L1+L2 | E4/E5 | 两者是否互补 |

#### E4：消融防作弊约束

**目的**：解决 E1 引入的作弊风险。

**实现方式**：

```python
# 方案 A：在 reward 计算中加惩罚（推荐，改动最小）
# nla_generate.py 中对部分样本（如 20%）额外生成一个零激活对照
# reward.py 中计算 cos_sim(explain_real, explain_zero)，过近则扣分

# 方案 B：在 AV SFT 阶段就加入对比数据
# 数据侧：增加一些"零激活 + 原始文本"的负样本
# 让 AV 学会：不同激活 → 不同解释
```

**推荐方案 A**，因为改动更小且在 RL 阶段实时生效。

**Warmup 策略**（来自 SAE 启发，详见第 8 节）：辅助 loss 权重必须从 0 线性 warmup 到目标值，防止训练初期多 loss 梯度冲突。
```python
warmup_steps = total_steps * 0.05  # 前 5% 步数
lambda_L1 = target_lambda_L1 * min(1.0, current_step / warmup_steps)
```

**消融检查**：
- 对比 E4 vs E1 中，零激活时 AV 解释的差异
- 理想情况：零激活 → AV 输出类似"无特定语义"的解释

### 4.4 阶段三：进阶 Loss（高成本，视情况推进）

| 实验 | 输入 | Loss 改动 | 对比基线 | 主要验证 |
|------|------|----------|---------|---------|
| E7 | P1 最优 | +L3（注意力一致性） | P1+L1 | 注意力约束是否有额外信息 |
| E8 | P1 最优 | +L1+L2+L3 | E7 | 全量组合 |

**注意**：L3 需要在 Stage 0 提取时就存储 attention weights，数据管线改动较大。建议在 P1/P2 验证有效后再考虑。

### 4.5 阶段四：训练范式变革（架构级改动，长期方向）

以上 P1-P3 的实验都是在现有 SFT+RL 框架内的改进。但我们也应探索**绕过 RL** 的根本性方案：

#### E9：Gumbel-Softmax 连续松弛

**核心思路**：AV 生成文本时，用 Gumbel-Softmax 替代离散采样，使整个 AV→AR→MSE 管线端到端可微，彻底去掉 RL。

```
# 当前流程（梯度在采样处断裂）
AV logits → argmax/sample → 离散 token → AR → MSE
              ↑ 不可微，梯度断

# Gumbel-Softmax 流程（梯度畅通）
AV logits → Gumbel-Softmax(τ) → soft token embeddings → AR → MSE
              ↑ 可微近似，τ→0 时逼近 argmax
```

**具体实现**：
1. AV 的 LM head 输出 logits 后，不取 argmax，而是用 Gumbel-Softmax 生成 soft one-hot
2. soft one-hot 与 embedding table 做加权求和，得到连续的 token embedding
3. 这些 soft embeddings 作为 AR 的输入
4. MSE loss 的梯度可以一路回传到 AV 的参数

**训练策略**：
- **温度退火**：τ 从 1.0 线性/指数退火到 0.1，初期 soft（探索），后期 hard（逼近真实）
- **Straight-Through Estimator (STE)**：前向用 hard token，反向用 soft 梯度
- **混合训练**：前期用 SFT（行为克隆），中后期切到 Gumbel-Softmax 端到端

**预期收益**：
- 消除 RL 的训练不稳定性和超参敏感度
- 梯度信号更直接，收敛更快
- 可以和 P1-P3 的所有输入/Loss 优化叠加

**风险**：
- soft tokens 和 hard tokens 存在 train-test mismatch
- 温度退火策略需要仔细调参
- 需要修改 AV 的采样逻辑和 AR 的输入方式，架构改动较大

**代码改动**：
- `nla/rollout/nla_generate.py`：AV 生成时用 Gumbel-Softmax 替代 `sample/greedy`
- `nla/rollout/nla_generate.py`：AR 输入从 token ids 改为 soft embeddings
- `nla/loss.py`：去掉 RL loss，只保留 MSE（此时可端到端优化）
- `configs/`：新增 Gumbel-Softmax 相关超参（初始温度、退火策略）

#### E10：Gumbel-Softmax + 最优输入/Loss 组合

在 E9 验证可行后，叠加 P1-P3 的最优组合。

### 4.6 实验优先级总结

```
P1 (先做):
  E0 → E1 → E2 → E3
  目标: 确定最佳输入侧组合

P2 (P1 完成后):
  E4 → E5 → E6
  目标: 确定最佳 Loss 组合，特别是防作弊约束

P3 (视 P2 结果决定):
  E7 → E8
  目标: 进阶优化，收益可能有限但值得一试

P4 (长期方向，可与 P1-P3 并行探索):
  E9 → E10
  目标: 训练范式变革，用 Gumbel-Softmax 消除 RL
```

---

## 5. 评估指标体系

### 5.1 自动指标（训练过程中直接观测）

| 指标 | 越高/低越好 | Baseline 参考 | 计算位置 |
|------|------------|-------------|---------|
| FVE (nrm) | 越高越好 | ≈ 0.72 (Qwen7B L20) | `nla/loss.py` |
| FVE (nrm_meannorm) | 越高越好 | ≈ 0.94 (Qwen7B L20) | `nla/loss.py` |
| Direction MSE | 越低越好 | 待测 | `nla/reward.py` |
| Cosine Similarity | 越高越好 | 待测 | `nla_inference.py` |

### 5.2 人工评估（最终判断）

自动指标只衡量"AR 能不能从文本重建激活"，不衡量"解释是否人类可读/准确"。

**评估流程**：
1. 随机抽取 50 个激活向量，每个方法生成解释
2. 人工打分（1-5 分）：
   - 准确性：解释是否准确描述了该位置激活的语义
   - 完整性：解释是否涵盖了关键信息
   - 无幻觉：解释是否包含激活中未编码的信息
3. 盲评：不告诉评分者哪个是哪个方法

### 5.3 消融检查指标（辅助验证）

| 检查 | 方法 | 目的 |
|------|------|------|
| 激活依赖性 | 对比真实激活 vs 零激活时 AV 的解释差异 | 验证 AV 确实在看激活而非抄文本 |
| 输入文本依赖性 | 对比有/无输入文本时 AV 的解释质量差异 | 验证输入文本的贡献 |
| 层号一致性 | 同一解释在不同层号下的差异 | 验证 AV 是否利用了层号信息 |

---

## 6. 代码改动清单

### 6.1 E1：给 AV 加输入文本

**文件 1：`nla/datagen/stage3_build.py`**

- `_PROVENANCE_COLS`：加入 `"detokenized_text_truncated"`（或通过 `--keep-debug-metadata` 已包含）
- 确保 RL parquet schema 中包含该列

**文件 2：`nla/datagen/stage3_build.py`**

- `_DEFAULT_ACTOR_TEMPLATE`：增加输入文本占位符
  ```
  ...Your overall task is to describe the semantic content of that activation vector.
  
  The original input text leading to this activation is:
  <input_text>{input_text}</input_text>
  
  Here is the vector:
  <concept>{injection_char}</concept>
  ...
  ```
- 需要新增 `{input_text}` 占位符，在 stage3_build 时填充

**文件 3：`nla/data_source.py`**

- 确认 `detokenized_text_truncated` 被正确加载到 `sample.metadata`
- 在 `_INJECT_PLACEHOLDER` 替换的同时，处理新的占位符

**文件 4：`nla/rollout/nla_generate.py`**

- `_prep_payload_sync`：构建 messages 时，将 `detokenized_text_truncated` 纳入 prompt

**文件 5：`nla/datagen/sidecar.py` / `nla/config.py`**

- 更新 sidecar schema 以包含新的 actor 模板信息
- 更新 neighbor token ID 计算

### 6.2 E2：给 AV 加层号

**文件：`nla/datagen/stage3_build.py`** 和 **`nla/rollout/nla_generate.py`**

- 在 AV prompt 中追加层号信息
- `activation_layer` 已在 parquet 中，只需在构建 prompt 时注入

### 6.3 E3：给 AV 加输出分布

**文件 1：`nla/datagen/stage0_extract.py`**

- 在 `extractor.extract()` 后，额外计算每个提取位置的 top-k 输出 token 及概率
- 在 parquet schema 中增加 `top_output_tokens` 列

**文件 2：`nla/datagen/stage3_build.py`**

- 将 `top_output_tokens` 列带入 AV-SFT 和 RL parquet

**文件 3：`nla/rollout/nla_generate.py`**

- 构建 prompt 时注入输出 token 信息

### 6.4 E4：消融防作弊约束

**文件 1：`nla/reward.py`**

- 在 `_prep_batch` 或 `_mse_to_reward` 中加入消融惩罚计算
- 需要在 rollout 时存储额外的零激活对照样本

**文件 2：`nla/rollout/nla_generate.py`**

- 对部分样本（20%）额外生成零激活对照的解释

---

## 7. 风险与应对

### 7.1 AV 抄捷径问题

**风险**：AV 有输入文本后，可能直接从文本猜解释，不看激活向量。

**检测方法**：
- 零激活测试：将激活向量替换为零向量，AV 的解释是否发生显著变化
- 如果变化不大 → AV 在抄文本

**应对**：
- L1 消融防作弊约束
- 输入文本做信息瓶颈（只给最后 N 个 token，或压缩后的摘要）
- 输入文本加噪声或 dropout

### 7.2 多 Loss 平衡问题

**风险**：多个 loss 量级不同、梯度方向可能冲突。

**应对**：
- 渐进式引入：先只训练 MSE，稳定后再逐个加入辅助 loss
- 动态权重：用 Uncertainty Weighting（Kendall et al. 2018）自动学习 λ
- 梯度归一化：GradNorm，确保各 loss 的梯度量级相当

### 7.3 计算成本问题

**风险**：注意力权重、梯度归因需要额外的前向/反向传播。

**应对**：
- 阶段三的实验暂不实施，待阶段一/二验证有效后再考虑
- 可以用离线预计算的方式（在数据生成阶段提取，而非训练时实时计算）

### 7.4 Context Window 溢出

**风险**：AV 的 prompt 增加输入文本、层号、输出分布后，可能占用过多 context window。

**应对**：
- 输入文本截断：只取最后 200 个 token
- 输出分布精简：只给 top-3 token
- 调整 `rollout_max_response_len` 平衡输入/输出长度

---

## 8. SAE 领域的启发与借鉴

我们系统调研了 SAE（Sparse Autoencoder）领域的最新进展，寻找对 NLA 实验设计的启发。

### 8.1 SAE 与 NLA 的本质区别

| 维度 | SAE | NLA |
|------|-----|-----|
| 中间表示 | 稀疏向量 `[0, 0, 3.7, 0, 0.1, ...]` | 自然语言 `"model is reasoning about safety"` |
| 可微性 | ✅ 全程可微（连续向量） | ❌ 离散采样处梯度断裂 |
| 训练范式 | 纯监督（MSE + L1 稀疏） | SFT（行为克隆）+ RL（策略梯度） |
| 信息容量 | 低（稀疏向量，受 L0 约束） | 高（自然语言，信息密度大） |
| 可解释性 | 需要事后标注 | 天然可解释（就是文本） |

**关键洞察**：SAE 的优化集中在"更好的稀疏化"（因为中间表示是低维向量），而 NLA 的优化方向应该是"更好的信息利用"（因为中间表示是高容量文本）。

### 8.2 SAE 架构优化对 NLA 的启发

#### 8.2.1 TopK SAE → 控制"解释的聚焦度"

TopK SAE 的核心改进：只保留最大的 k 个特征激活，其余归零。直接控制 L0 稀疏度，无需调 L1 系数。

**对 NLA 的启发**：NLA 中没有直接的稀疏性约束，但可以类比——**AV 的解释是否应该聚焦在核心语义上？**

可能的实现：
- 在 AV 的 prompt 中加"用一句话概括核心语义"的指令
- 或限制解释文本的最大长度（间接的"稀疏"约束）
- 或加一个辅助 loss 惩罚过长/过散的解释

**优先级**：低。当前 NLA 的解释文本已经较短，过度压缩可能损失信息。

#### 8.2.2 Gated SAE → 分离"识别"与"量化"

Gated SAE 的核心改进：将"是否激活"（门控，0/1）和"激活多少"（幅度，连续值）分开建模，解决了 SAE 的 shrinkage 问题（激活值被系统性地低估）。

**对 NLA 的启发**：解释文本可以**分离"什么概念"和"强度如何"**。

```
# 当前解释
"the model is processing information about danger and risk assessment"

# Gated 式解释
"Concept: danger/risk assessment (confidence: high, layer: 20/32)"
```

可能的实现：
- 修改 AV 的 prompt 模板，要求分两部分输出
- 或训练一个辅助的"强度头"预测激活的范数

**优先级**：中。有理论价值，但改变了 AV 的输出格式，需要在 P1-P3 完成后考虑。

#### 8.2.3 JumpReLU / BatchTopK → 直接优化目标而非代理

JumpReLU SAE 直接优化 L0（活跃特征数），而非 L1（L1 是 L0 的凸代理，但会引入偏差）。BatchTopK 在 batch 级别做 top-k，允许不同样本有不同的稀疏度。

**对 NLA 的启发**：我们是否也在用代理指标而非直接优化目标？

当前 NLA 的训练信号链：
```
真正目标: 解释文本准确描述激活语义（不可直接优化）
  ↓ 代理
MSE 重建误差（可优化，但不完全等价于真正目标）
  ↓ 再代理
RL reward 基于 MSE（又引入了策略梯度的方差）
```

我们实际上在用两层代理。**Gumbel-Softmax（E9）去掉了一层代理（去掉 RL，直接用 MSE）。** 但 MSE 本身仍然是代理——好的 MSE 不一定意味着好的解释。

**对评估的启发**：最终应引入人类评估或更强的自动评估指标（如概念匹配度），而非仅依赖 MSE/FVE。

### 8.3 SAE 训练技巧对 NLA 的启发

Anthropic 2024.04 公布的 SAE 训练最佳实践：

| 技巧 | SAE 中的做法 | NLA 中的对应 | 是否已用 |
|------|------------|------------|---------|
| Decoder 列不再约束为单位范数，改为 L1×‖W‖₂ | 简化训练，用惩罚替代约束 | NLA 没有类似的约束 | 不适用 |
| 数据集缩放到 E[‖x‖] = √n | 统一不同层的激活范数 | 当前 NLA 使用 `mse_scale` 做归一化 | ✅ 已有 |
| λ 线性 warmup（前 5% steps） | 稀疏惩罚从 0 渐增 | 辅助 loss（L1-L4）应做 warmup | ❌ **应加** |
| LR 线性衰减（最后 20%） | 稳定收敛 | 当前 NLA 使用 cosine decay | ✅ 已有 |
| 不再使用 resampling/ghost grads | 简化训练 | NLA 没有死特征问题 | 不适用 |

**关键启发：辅助 Loss 的 Warmup 策略**

SAE 的经验表明，稀疏惩罚从 0 逐渐增加到目标值（而非一开始就用全量惩罚）能显著稳定训练。这对我们的 P2 阶段（加入 L1-L4 辅助 loss）至关重要：

```python
# SAE 的做法：λ 线性 warmup
lambda_t = target_lambda * min(1.0, t / warmup_steps)

# NLA 应该效仿：辅助 loss 权重 warmup
lambda_L1_t = target_lambda_L1 * min(1.0, t / warmup_steps)  # 防作弊约束
lambda_L2_t = target_lambda_L2 * min(1.0, t / warmup_steps)  # 行为一致性
```

### 8.4 Attribution Dictionary Learning → 直接验证我们的 L4

Anthropic 的 Attribution Dictionary Learning 是与我们 L4（输入归因一致性 loss）最直接对应的工作。

**他们的方法**：
```
L = ||x - x̂||²              重建误差
  + λ||y||₁                  激活稀疏惩罚（原有）
  + α||A_y||₁                归因稀疏惩罚（新增）
  + β|ΣA_{x-x'}|            未解释归因惩罚（新增）
```
其中 `A_y = y ⊙ ∇_y L_LLM`（特征激活对 LLM loss 的归因）。

**他们的结论**："At first glance, they seemed about equally good"——初步实验未能显著优于标准 SAE。

**对我们的启发**：
1. **L4 的效果可能有限**——Anthropic 已经初步尝试，效果不明显。我们将 L4 优先级降低是正确的。
2. **但不应放弃**——Anthropic 的实验是在 SAE（稀疏向量）上做的，而 NLA 的中间表示是自然语言，信息容量远大于稀疏向量。归因信息在 NLA 中的价值可能更高，因为文本可以利用更丰富的语义约束。
3. **实现方式可借鉴**——他们用 `y ⊙ ∇_y L_LLM` 计算归因，比完整的梯度回传高效得多。我们可以用类似的方式为 L3/L4 高效计算归因信号。

### 8.5 Gurnee 的发现 → 支持 L2（行为一致性）

Gurnee (2024) 发现 SAE 的重建误差是"病态的"：**小的 MSE 重建误差不保证下游行为一致**。一个 SAE 可以很好地重建激活向量，但在功能上与原始激活等价性很差。

**对我们的启发**：
1. **MSE 不是完美指标**——它衡量的是向量层面的接近度，不保证语义/行为层面的等价。这支持了我们的 L2（行为一致性 loss）。
2. **FVE 高不等于解释好**——即使 AR 完美重建激活向量，也不能保证解释文本准确。最终仍需人工评估或更强的自动评估。
3. **但 MSE 仍是必要条件**——MSE 差则解释一定差，只是 MSE 好不能保证解释好。

### 8.6 SAE 领域的整体判断

| 启发 | 对应我们的优化 | 优先级调整 |
|------|--------------|-----------|
| TopK：控制稀疏度 | 解释文本聚焦度 | 低，暂不加入 |
| Gated：分离门控与幅度 | 概念+强度分离输出 | 中，P3 之后考虑 |
| JumpReLU：直接优化 L0 | 评估指标应更直接 | 已在评估体系中反映 |
| 辅助 loss warmup | L1-L4 的 warmup 策略 | **高——应加入 P2 实验设计** |
| Attribution DL：归因稀疏 | L4 输入归因 | 低——Anthropic 未验证有效 |
| Gurnee：MSE 病态 | L2 行为一致性 | 中——MSE 不够但不是首要瓶颈 |
| SAE 无额外信息利用 | I1-I3 输入侧优化 | **高——这是 NLA 独有的优势** |

**最核心的启发**：SAE 领域没有人探索过"利用输入文本、输出分布等额外语义信息"这个方向，因为 SAE 的中间表示是低维稀疏向量，塞不进这些信息。而 NLA 的中间表示是自然语言，天然可以承载丰富的语义约束。**这是 NLA 相对于 SAE 的核心差异化优势，也是我们实验的最大创新点。**

---

## 附录：关键数据结构

### Parquet Schema（Stage 3 输出）

```
AV-SFT:
  prompt: list[dict]  # [{"role":"user","content": actor_template}]
  response: str       # "<explanation>\n{api_explanation}\n</explanation>"
  activation_vector: list[float]  # RAW, d_model 维
  n_raw_tokens: int
  activation_layer: int
  doc_id: str
  detokenized_text_truncated: str  # 仅 --keep-debug-metadata

AR-SFT:
  prompt: str         # critic_template 填充后的完整字符串
  activation_vector: list[float]
  n_raw_tokens: int
  activation_layer: int
  doc_id: str

RL:
  prompt: list[dict]  # 同 AV-SFT 但无 response
  activation_vector: list[float]
  n_raw_tokens: int
  activation_layer: int
  doc_id: str
```

### Sidecar YAML 关键字段

```yaml
kind: nla_dataset
stage: rl  # or av_sft, ar_sft
extraction:
  base_model: Qwen/Qwen2.5-7B
  d_model: 3584
  layer_index: 20
  norm: none
tokens:
  injection_char: ㊗
  injection_token_id: 1234
  injection_left_neighbor_id: 5678
  injection_right_neighbor_id: 9012
  critic_suffix_ids: [...]
prompt_templates:
  actor: "You are a meticulous AI researcher..."
  critic: "Summary of the following text: ..."
```

### 多模态训练输入键

```python
MM_ACTIVATION_KEY = "nla_activation"      # [1, d] tensor, RAW
MM_CRITIC_TOKENS_KEY = "nla_critic_tokens" # [T] tensor, tokenized critic prompt
MM_MSE_SCALE_KEY = "nla_mse_scale"        # float, normalization scale
```
