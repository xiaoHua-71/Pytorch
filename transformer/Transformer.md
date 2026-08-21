# Transformer 到 Agent 应用开发学习笔记

> 目标：看完这份笔记后，你应能回答三个问题：**Transformer 怎样把文本变成下一个词？大模型为什么能遵循指令？一个 Agent 如何可靠地调用工具完成任务？**

---

## 0. 先建立全局地图

可以把学习路线看成一条数据流：

```text
文本 -> Token -> 向量 + 位置信息 -> Transformer -> 下一个 Token 概率
                                      |
                                      v
                         预训练 -> 指令微调 -> 对齐
                                      |
                                      v
用户问题 -> LLM(推理/决策) -> 工具(RAG、搜索、数据库、代码) -> 观察结果
                    ^                                           |
                    +-------------- 循环直到完成 ---------------+
```

前半段是“模型怎么工作”，后半段是“怎样让模型在真实软件中完成任务”。

### 需要的基础

- Python：函数、类、列表/字典、异常处理。
- PyTorch：`Tensor`、shape、矩阵乘法、训练循环。建议先完成仓库中线性回归、RNN 的笔记。
- 数学：向量/矩阵乘法、softmax、导数的直观含义。无需一开始推完所有公式。

**学习原则：先能画出数据的 shape，再背公式。** 例如 batch 为 `B`、序列长度为 `T`、隐藏维度为 `d_model` 时，词向量张量通常是 `[B, T, d_model]`。

---

## 1. Transformer 解决了什么问题？

在 Transformer 之前，常用 RNN/LSTM 按顺序读词：读到第 `t` 个词时依赖前面的隐藏状态。这有两个问题：

1. 很难并行训练，句子必须一个词一个词地处理。
2. 长距离信息会在反复传递中衰减，例如“我把书放在桌上，因为**它**很重”中的“它”指谁。

Transformer 的核心想法是：**让每个词直接查看句中所有相关词，并为不同词分配不同权重。** 这叫 Attention（注意力）。它能并行计算，也更容易建立远距离关系。

### 代表论文 1：Attention Is All You Need（2017）

- 论文：[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- 贡献：提出纯 Attention 的 Encoder-Decoder 架构，即 Transformer。
- 你要抓住的不是所有细节，而是它的模块化结构：Embedding、位置编码、多头注意力、前馈网络、残差连接、LayerNorm。

---

## 2. 从文本到向量

神经网络只能处理数字。文本进入模型前要经历以下过程：

```text
“我喜欢机器学习”
  -> tokenizer
[“我”, “喜欢”, “机器”, “学习”] 或子词 token id
  -> embedding lookup
[[...], [...], [...], [...]]
  -> 加上 position embedding
模型输入 X: [T, d_model]
```

### 2.1 Tokenizer：不是按“字”或“词”那么简单

主流 LLM 使用子词切分（BPE、WordPiece 或 Unigram）。常见词可能是一个 token，罕见词会拆成多个片段。模型真正预测的是 **token**，不是汉字或单词。

- 输入上限和 API 价格通常按 token 计。
- 同一句话在不同模型上的 token 数可能不同。
- Agent 做 RAG 时，要按 token 长度切文档，而不是机械按字符数切。

### 2.2 Embedding：token 的可学习坐标

词表大小为 `V`，隐藏维度为 `d_model`，Embedding 是矩阵 `E`，shape 为 `[V, d_model]`。token id `i` 查表得到第 `i` 行向量。

相近语义的 token 在高维空间中通常更接近，但不要把它理解为绝对的“语义字典”：真正的含义还会随上下文改变。

### 2.3 位置信息：注意力本身不知道顺序

若只看 token 向量，“我爱你”和“你爱我”只是同一组向量。Transformer 因此必须加位置：

```text
输入 = Token Embedding + Position Embedding
```

原论文使用正弦/余弦位置编码；现代模型也常用可学习位置编码、RoPE（旋转位置编码）等。初学阶段只需知道：**位置编码告诉模型“第几个 token”，注意力告诉模型“应关注谁”。**

---

## 3. 注意力：Transformer 的发动机

对每个输入向量 `X`，通过三组可学习矩阵得到：

```text
Q = X W_Q  (Query：我正在找什么)
K = X W_K  (Key：我包含什么线索)
V = X W_V  (Value：若关注我，应取走什么信息)
```

缩放点积注意力公式：

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

按三步理解即可：

1. `Q K^T`：每个词与每个词算相关性分数，shape 为 `[T, T]`。
2. `softmax`：将每一行分数变成和为 1 的权重。
3. 权重乘 `V`：每个词汇总它关注的词的信息。

`/ sqrt(d_k)` 是为了避免维度大时分数过大，使 softmax 过于极端、训练不稳定。

### 一个直觉例子

处理“苹果公司发布了新手机”中的“苹果”时，模型可对“公司”“手机”赋予较高注意力，从而倾向理解为公司而非水果。注意力权重是动态计算的，所以同一个 token 在不同句子里可以有不同表示。

### Multi-Head Attention：一次从多个角度看

把隐藏维拆成多个 head，每个 head 有自己的 `W_Q/W_K/W_V`。有的 head 可能更关注邻近词，有的更关注主谓关系或指代关系，最后将各 head 拼接并投影。

```text
MultiHead(X) = Concat(head_1, ..., head_h) W_O
```

多头不等于每个头都有可解释的固定职责，但它增加了模型表示不同关系的能力。

### Mask：决定“允许看哪里”

- **Padding mask**：忽略为补齐长度而加入的 `<pad>`。
- **Causal mask（因果掩码）**：第 `t` 个位置只能看自己和过去，不能偷看未来答案。

因果掩码使 GPT 能按以下方式生成：

```text
“今” -> 预测 “天” -> “今天” -> 预测 “天” -> “今天天” ...
```

实际采样会结合上下文和概率分布，不是每次都简单选概率最高的 token。

---

## 4. 一个 Transformer Block 长什么样？

典型 Block 有两层子模块：注意力层和逐位置前馈网络（FFN）。每层外面都有残差连接与归一化。

```text
X
 -> Multi-Head Attention
 -> Add & Norm                 # 残差：保留原信息，帮助深层训练
 -> FFN                        # 对每个位置独立地做非线性变换
 -> Add & Norm
 -> 输出
```

FFN 常写作：`FFN(x) = activation(x W_1 + b_1) W_2 + b_2`。它不负责 token 间交流（那是 Attention 的工作），而是加工每个 token 聚合后的特征。

残差连接可概括为 `y = x + f(x)`：即使新层没有学好，信息仍能沿 `x` 向后传播。LayerNorm 则让每层输入的数值尺度更稳定。

---

## 5. 三种 Transformer 家族

| 类型 | 可看见的上下文 | 典型模型 | 擅长任务 |
| --- | --- | --- | --- |
| Encoder-only | 双向，能看全句 | BERT | 分类、检索、向量表示 |
| Decoder-only | 只能看左侧历史 | GPT、Llama、Qwen | 续写、对话、工具调用 |
| Encoder-Decoder | Encoder 看全句，Decoder 自回归生成 | T5、BART | 翻译、摘要、文本到文本 |

### 代表论文 2：BERT（2018）

- 论文：[Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
- 方法：随机遮住输入中的部分 token，让 Encoder 猜回来（Masked Language Modeling）。
- 启发：Encoder 能生成很适合“理解”和“相似度检索”的上下文向量。

### 代表论文 3：GPT-3（2020）

- 论文：[Brown et al., 2020](https://arxiv.org/abs/2005.14165)
- 方法：只做“预测下一个 token”的大规模预训练，通过提示词进行 few-shot 学习。
- 启发：Decoder-only 的简单目标，在足够数据和参数下可泛化到大量任务。这是今天聊天大模型和 Agent 的主要底座。

---

## 6. 大模型如何训练和生成

### 6.1 预训练：学习语言与世界知识

给定 token 序列 `[x_1, x_2, ..., x_T]`，Decoder 模型学习：

```text
P(x_1, ..., x_T) = Π P(x_t | x_<t)
```

训练时答案已知，模型同时预测所有位置的下一个 token，并用交叉熵损失让正确 token 的概率变大。这个阶段消耗大量数据和算力，个人通常不从零训练。

### 6.2 指令微调（SFT）：学会“按要求回答”

使用高质量的 `(指令, 理想回答)` 数据继续训练。预训练模型会续写，SFT 后才更像助手：知道回答格式、角色、边界和任务目标。

### 6.3 偏好对齐：让回答更符合人类偏好

常见思路是收集同一问题的两个回答，由人或模型标出更好的一个，再优化模型偏好。你会看到 RLHF、DPO 等术语。

- RLHF：[InstructGPT](https://arxiv.org/abs/2203.02155)，用人类反馈训练奖励模型并强化学习。
- DPO：[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)，以更直接的方式利用偏好对。

初学 Agent 时不必自己训练对齐模型，但要明白：**模型的“乐于助人”和“遵循指令”主要不是 Transformer 结构天然带来的，而是训练数据与对齐过程带来的。**

### 6.4 推理（inference）：每次只产生一个 token

生成流程：输入 prompt -> 得到词表 logits -> 选一个 token -> 拼回输入 -> 重复，直到结束标记或长度上限。

常见采样参数：

- `temperature`：越低越保守、稳定；越高越发散、富有随机性。
- `top_p`：只从累计概率达到 `p` 的候选集合中采样。
- `max_tokens`：限制本次输出长度。

对需要正确执行工具的 Agent，通常使用较低温度；对头脑风暴或生成多个方案，可适度提高。

---

## 7. 从 LLM 到 Agent：多了什么？

LLM 本身根据上下文生成 token；**Agent 是一个由程序控制的闭环系统**，让 LLM 负责语言理解与决策，让外部系统负责事实、计算和行动。

```text
用户：查本周销售额，并与上周比较
   |
LLM：判断需要数据库工具，生成结构化参数
   |
程序：校验参数并执行 SQL/API
   |
工具结果：{this_week: 120000, last_week: 100000}
   |
LLM：解释“增长 20%”，必要时继续调用工具
   |
用户得到最终答案
```

一个可用 Agent 通常由以下部分组成：

| 组件 | 负责什么 | 最小实现 |
| --- | --- | --- |
| Model | 理解、规划、生成 | 一个 Chat Completions/Responses 调用 |
| Instructions | 角色、边界、输出规则 | system prompt |
| Tools | 访问外部能力 | Python 函数、HTTP API、数据库查询 |
| Memory/State | 保存本轮或跨轮状态 | 对话历史、数据库、向量库 |
| Orchestrator | 执行工具调用循环 | 你的 Python 代码/Agent 框架 |
| Guardrails | 权限、输入输出安全 | schema、白名单、人工确认 |
| Evals/Tracing | 检查质量、定位失败 | 测试集、调用日志、span trace |

关键边界：**模型不能直接运行代码、发送邮件或读数据库。它只会产出文本/结构化调用意图；真正有权限的永远是你的应用程序。**

---

## 8. Agent 的三种核心能力

### 8.1 Tool Calling：让输出变成可靠的程序输入

不要让模型自由生成“请调用天气接口，城市是北京”这样的自然语言，然后靠字符串解析。应定义函数名、描述和 JSON Schema，让模型返回结构化参数，例如：

```json
{
  "name": "get_weather",
  "arguments": {"city": "北京", "date": "2026-08-20"}
}
```

你的程序必须做三件事：

1. 根据 schema 校验类型、必填字段、枚举值。
2. 再做业务权限校验，例如用户能否访问该数据。
3. 执行工具，将结果作为 tool message 放回模型上下文。

**Schema 减少格式错误，不保证业务正确或安全。** SQL、文件路径、金额、收件人等高风险参数还必须由代码限制。

### 8.2 RAG：给模型“查资料”，而不是“背资料”

RAG（Retrieval-Augmented Generation）适合私有文档、频繁变化的知识和需要出处的问答。

```text
文档 -> 清洗/切块 -> Embedding -> 向量数据库
用户问题 -> Embedding -> Top-k 检索 -> (问题 + 证据)交给 LLM -> 带引用回答
```

最容易忽视的事实：RAG 的难点通常不是“选哪个向量库”，而是文档解析、切块、元数据、检索评测与引用展示。

实用起点：每块约 300-600 tokens，保留标题/文件名/页码等 metadata；先用关键词检索 + 向量检索，再考虑 reranker。回答必须基于检索到的片段；证据不足时应明确说不知道。

### 8.3 ReAct：推理与行动交替

### 代表论文 4：ReAct（2022）

- 论文：[Yao et al., 2022](https://arxiv.org/abs/2210.03629)
- 思路：模型在“思考/决定下一步”和“调用行动、观察结果”之间交替。

简化的执行循环：

```python
messages = [system_message, user_message]
for _ in range(MAX_STEPS):
    response = llm(messages, tools=TOOLS)
    messages.append(response)

    if not response.tool_calls:
        return response.text

    for call in response.tool_calls:
        args = validate(call.arguments)
        result = execute_allowed_tool(call.name, args)
        messages.append(tool_message(call.id, result))
raise RuntimeError("agent exceeded step limit")
```

真实项目还要设置超时、重试策略、最大步数、总 token/费用上限和审计日志。不要无限循环，也不要把内部思维过程当作产品依赖；以可观察的“工具调用、状态、结果”为调试对象。

---

## 9. 用论文理解 Agent 的演进

| 论文 | 核心问题 | 可迁移到工程的结论 |
| --- | --- | --- |
| [RAG, 2020](https://arxiv.org/abs/2005.11401) | 如何引入外部知识 | 知识更新优先走检索，不要急着微调 |
| [WebGPT, 2021](https://arxiv.org/abs/2112.09332) | 如何浏览网页并引用来源 | 工具结果要保留来源，答案应可追溯 |
| [ReAct, 2022](https://arxiv.org/abs/2210.03629) | 如何把推理和行动结合 | 采用“模型决策 -> 工具观察 -> 再决策”的循环 |
| [Toolformer, 2023](https://arxiv.org/abs/2302.04761) | 模型何时、怎样使用 API | 工具描述和返回结果的格式是产品能力的一部分 |
| [Reflexion, 2023](https://arxiv.org/abs/2303.11366) | 如何从失败中改进 | 将失败案例存入评测集；不要无边界地把反思写入长期记忆 |
| [MRKL, 2022](https://arxiv.org/abs/2205.00445) | 如何路由到不同专家模块 | 用路由器/工具选择处理计算、搜索、数据库等专长任务 |

读论文的正确姿势：先读 Abstract、Introduction、图 1、实验结论，写下“问题、方法、假设、局限、我项目中的一个用法”，最后再看方法细节。不要第一次就试图读懂每个证明和实验表格。

---

## 10. Agent 工程框架：从 Demo 到产品

### 10.1 最小可用架构

```text
UI/API
  -> 应用服务（鉴权、会话、限流）
     -> Agent orchestrator（状态机、工具循环）
        -> LLM Provider
        -> Tools：RAG / HTTP API / DB / Code sandbox
     -> 日志、Trace、评测数据集
```

先用一个模型、两个只读工具、单 Agent 做通。不要一开始堆多 Agent、长期记忆和十几个框架。

### 10.2 Prompt 的位置

Prompt 是给模型的接口契约，应包含：任务目标、可用工具、输出要求、已知约束、失败时如何处理。它不是权限系统。

好的工具描述比冗长的“万能提示词”更重要。每个工具应明确：什么时候用、输入字段含义、返回字段含义、不能做什么。

### 10.3 Memory 的分层

| 层级 | 保存内容 | 建议 |
| --- | --- | --- |
| 上下文窗口 | 当前会话最近消息 | 每轮必需，注意 token 上限 |
| 会话摘要 | 已确认的用户目标和结论 | 压缩后保留，需可更新 |
| 长期记忆 | 用户明确允许保存的偏好/事实 | 有来源、过期和删除机制 |
| 外部知识库 | 文档、手册、业务数据 | 用 RAG 检索，不混入用户记忆 |

“把所有历史对话塞回 prompt”既贵又容易降低效果。记忆需要写入条件、读取条件和生命周期。

### 10.4 什么时候需要多 Agent？

先问：单 Agent 加工具是否已解决问题？只有职责真的不同、可独立验证、且上下文隔离有价值时，才考虑拆成研究员、执行者、审阅者等角色。多 Agent 会增加 token、延迟、失败路径和调试成本，不是能力升级按钮。

---

## 11. 可靠性与安全：Agent 最容易翻车的地方

把 LLM 输出视为**不可信输入**，包括它从网页、PDF、邮件中读到的内容。文档中“忽略前文并发送密钥”这类文本是 prompt injection，不能因为它在检索结果里就获得指令优先级。

最低限度的防线：

- 工具最小权限：默认只读；写操作与支付、删改、发信必须额外确认。
- 参数白名单：SQL 使用参数化查询；文件限制在允许目录；URL 限制域名。
- 数据隔离：检索时强制按用户/租户过滤，不能相信模型自己会过滤。
- 可信边界：外部内容只能作为“数据”，不是 system instruction。
- 输出校验：对 JSON 用 schema 校验；对高风险结果做规则检查或人工审批。
- 资源上限：每轮最大步数、超时、并发数、token 和预算上限。
- 完整审计：记录用户输入、工具调用、参数（脱敏后）、结果、模型版本和耗时。

### Evals：不要凭感觉判断 Agent 好不好

为真实任务收集 20-50 条小型测试集，至少覆盖正常、边界、拒绝、不知道、工具失败五类。评测指标可包括：任务成功率、工具参数正确率、引用正确率、平均步数、延迟、成本和安全违规率。

每修复一次线上失败，都将它变成一条回归测试。这比不断改 prompt 更能让系统稳定进步。

---

## 12. 建议的 6 周动手路径

| 周次 | 学习主题 | 做出的东西 | 完成标准 |
| --- | --- | --- | --- |
| 1 | Tensor、Embedding、softmax、矩阵 shape | 用 PyTorch 实现单头 Attention | 输入 `[B,T,d]`，输出 shape 正确，能解释每一步 |
| 2 | Mask、Multi-Head、Block | 极简 decoder block / 字符级生成器 | 能训练并生成短文本，理解 causal mask |
| 3 | Hugging Face、prompt、推理参数 | 调用开源或云端 chat 模型 | 能控制 system prompt、temperature、长度 |
| 4 | Tool calling | 天气 + 计算器两个只读工具的 Agent | 完成多轮工具循环，参数经过校验 |
| 5 | RAG | 对本仓库 Markdown 问答并返回文件出处 | 有切块、检索、引用和“证据不足”处理 |
| 6 | Evals、安全、部署 | 一个可演示的领域助手 | 有测试集、trace、权限边界和 README |

推荐第一个完整项目：**“PyTorch 学习助手”**。它检索当前仓库笔记，回答“Attention 的输入 shape 是什么”，必要时调用计算器验证参数量。这个项目的范围小，但覆盖了 RAG、tool calling、会话状态、引用、评测和安全边界。

---

## 13. 最常见的误区

1. **把 Attention 当作知识库。** Attention 处理当前上下文；长期、可更新知识应使用 RAG 或数据库。
2. **把 RAG 当作万能解。** 检索不到、文档过期、上下文太长时，模型仍会答错；必须评测召回和引用。
3. **认为 function calling 会让模型 100% 正确。** 它改善格式，不能代替权限校验与业务规则。
4. **为了“聪明”而上多 Agent。** 先把单 Agent 的状态、工具与评测做好。
5. **只调 prompt，不记录失败。** 没有 trace 和测试集，就无法知道改动是否真的变好。
6. **让 Agent 直接执行高权限动作。** 任何真实世界写操作都要权限、确认和审计。

---

## 14. 最终心智模型

```text
Transformer：一种用注意力处理序列的神经网络
LLM：经过预训练、指令微调与对齐的 Transformer（通常是 Decoder）
RAG：在回答前为 LLM 找到可引用的外部证据
Tool calling：让 LLM 提出结构化调用意图，由程序安全地执行
Agent：LLM + 状态 + 工具 + 执行循环 + 安全边界 + 评测/可观测性
```

学习时始终区分两件事：

- **模型能力**：理解上下文、生成语言、根据示例做模式归纳。
- **应用能力**：数据质量、工具设计、权限控制、流程编排、评测与用户体验。

前者决定“它能想出什么”，后者决定“它能否在现实中可靠地做成事”。

## 延伸阅读顺序

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：重点看结构图和第 3 节。
2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)：配合图理解注意力。
3. [BERT](https://arxiv.org/abs/1810.04805) 与 [GPT-3](https://arxiv.org/abs/2005.14165)：理解三类架构和预训练目标。
4. [InstructGPT](https://arxiv.org/abs/2203.02155)：理解为什么模型会听指令。
5. [RAG](https://arxiv.org/abs/2005.11401)、[ReAct](https://arxiv.org/abs/2210.03629)、[Toolformer](https://arxiv.org/abs/2302.04761)：连接到 Agent。

> 下一步：先亲手实现一个带 causal mask 的单头 Attention；在能打印并解释每个张量 shape 后，再进入大模型 API 和 Agent 框架。
