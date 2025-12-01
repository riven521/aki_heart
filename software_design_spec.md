# 《基于多模态异构数据的心脏外科术后急性肾损伤风险预测系统 V1.0》软件设计说明书

## 1. 设计目的（Design Objectives）
### 1.1 软件目的
- 提供针对心脏外科术后患者的急性肾损伤（Acute Kidney Injury, AKI）风险预测功能，生成风险概率与风险等级，辅助临床决策。
- 通过多模态异构数据（结构化临床指标、临床文本、术中时序信号）提升预测准确性和可解释性。

### 1.2 系统角色
- **临床研究人员/医生**：上传或录入患者术前、术中数据，查看风险预测结果。
- **数据管理员**：管理数据集版本、日志与历史记录，保证数据合规。
- **系统运维人员**：维护模型与服务运行状态，更新模型版本。

### 1.3 在临床研究流程中的位置
- 位于术后风险评估环节：在数据采集与清洗之后、临床决策支持之前，为术后 AKI 风险提供定量参考。
- 可嵌入临床研究数据管线，与电子病历系统或科研数据仓库对接，输出风险报告。

### 1.4 约束与假设
- 假设输入数据经过基本脱敏，符合隐私保护要求。
- 计算资源支持 PyTorch 推理（CPU/GPU 均可），FastAPI 后端部署于内网或受控环境。
- 前端通过 HTTPS 调用后端 API，浏览器需支持现代 JavaScript。
- 模型输入需符合定义的 schema，缺失值按预处理规则处理。

## 2. 总体设计（Overall System Design）
### 2.1 系统概述
系统由 Web 前端、FastAPI 后端、数据预处理与模型推理管道组成。前端负责数据录入与结果展示；后端负责接口、数据校验、模型调用与日志记录；模型管道由文本编码、时序编码、结构化数据 MLP、特征融合与风险评分构成。

### 2.2 模块分解
- Web UI 模块：数据输入表单、文本框、时序文件上传，结果展示。
- Backend API 模块（FastAPI）：路由、请求校验、调用推理服务、返回 JSON。
- 数据预处理与校验模块：字段校验、缺失值处理、标准化、时序对齐。
- 文本编码模块（BERT/Dummy Encoder）：将 note_text 转为固定维度向量。
- 时序编码模块（LSTM）：处理术中时序信号，输出隐藏状态向量。
- 结构化数据 MLP 模块：对临床结构化特征进行非线性变换。
- 多模态融合模块：使用拼接 + 注意力权重对各模态特征融合。
- 风险预测与评分模块：线性层 + Sigmoid 生成风险概率，阈值映射风险等级。

### 2.3 组件职责
- **UI 组件**：收集输入、展示风险评分与等级、提示异常。
- **API 控制器**：处理 POST /predict 请求，调用预处理与模型推理。
- **预处理器**：保证数据质量，统一数据格式，生成模型输入张量。
- **模型推理器**：调用 PyTorch 模型进行前向计算，输出概率与等级。
- **日志与存储**：记录请求、响应、错误与版本信息，可选存入数据库。

## 3. 系统架构图（System Architecture Diagram）
```
+-----------+      +-----------+      +----------------+      +-------------------+      +---------------+
|   Web UI  | ---> |  FastAPI  | ---> | Model Pipeline | ---> | PyTorch Inference | ---> | Result Output |
+-----------+      +-----------+      +----------------+      +-------------------+      +---------------+
```

## 4. 模块设计（Detailed Module Design）
### 4.1 Web UI 模块
- **功能说明**：提供数据输入表单（结构化字段、文本、时序上传），触发预测请求，展示风险结果与提示信息。
- **输入/输出**：
  - 输入：用户填报的 clinical_features（数值/分类字段）、note_text（文本）、timeseries（CSV/JSON 格式时序数据）。
  - 输出：风险概率 risk_score、风险等级（低/中/高）、提示消息。
- **数据结构**：
  - `clinical_features`: 键值对（数值或枚举字符串）。
  - `note_text`: 字符串。
  - `timeseries`: 以时间戳和多通道信号组成的列表或表格。
- **内部逻辑**：表单校验 → JSON 组装 → 调用 `/predict` → 解析返回 → 在页面展示概率与等级。
- **异常处理**：前端校验必填项；捕获网络异常显示用户友好提示；对 4xx/5xx 响应展示错误消息。

### 4.2 Backend API 模块（FastAPI）
- **功能说明**：定义 REST 接口，接收前端请求，调用预处理与模型推理，返回 JSON 结果。
- **输入/输出**：
  - 输入：POST `/predict`，Body 为 JSON {clinical_features, note_text, timeseries}。
  - 输出：JSON {risk_score, risk_category, model_version, timestamp}。
- **数据结构**：使用 Pydantic Schema 定义字段类型、必填性和校验规则。
- **内部逻辑**：请求校验 → 调用预处理器 → 模型推理 → 格式化响应 → 日志记录。
- **异常处理**：
  - 校验失败返回 422 + 错误详情；
  - 推理异常返回 500 + 错误描述；
  - 捕获超时/资源不足异常并记录日志。

### 4.3 数据预处理与校验模块
- **功能说明**：对三类数据进行格式、范围、缺失值校验与标准化，生成模型可接受的张量。
- **输入/输出**：
  - 输入：raw clinical_features、raw note_text、raw timeseries。
  - 输出：`structured_tensor`、`text_tensor`（token IDs/embeddings）、`timeseries_tensor`（对齐的时序矩阵）。
- **数据结构**：
  - 标准化后的 `structured_tensor`: shape (batch, d_struct)。
  - 文本 token 序列：shape (batch, seq_len)。
  - 时序张量：shape (batch, time_steps, channels)。
- **内部逻辑**：
  - 结构化：缺失填充、类别编码、数值标准化。
  - 文本：清理、截断/填充、token 化。
  - 时序：时间对齐、插值/填充、归一化。
- **异常处理**：
  - 检测缺失与非法值，记录并返回错误。
  - 对异常时间戳或不对齐数据抛出校验错误。

### 4.4 文本编码模块（BERT 或 Dummy Encoder）
- **功能说明**：将临床文本 note_text 转为固定维度向量；在资源受限场景可使用轻量 Dummy 编码器。
- **输入/输出**：
  - 输入：文本 token 序列（IDs）；
  - 输出：文本特征向量 `text_feat` (dim = d_text)。
- **数据结构**：Token IDs、attention mask；输出为张量。
- **内部逻辑**：
  - 若使用 BERT：通过预训练模型获取 [CLS] 向量或池化输出。
  - 若 Dummy：平均 token embedding 或简单 TF-IDF 映射。
- **异常处理**：模型加载失败、超长文本截断报警，记录日志。

### 4.5 时序编码模块（LSTM）
- **功能说明**：处理术中时序信号（生命体征/监护数据），提取时间依赖特征。
- **输入/输出**：
  - 输入：`timeseries_tensor` (batch, T, C)。
  - 输出：时序特征向量 `ts_feat` (dim = d_ts)。
- **数据结构**：浮点张量，支持可变长度通过 padding + mask。
- **内部逻辑**：双向或单向 LSTM 前向，取最终隐藏状态或池化（max/mean）作为表征。
- **异常处理**：处理长度为 0 的序列，给出错误；捕获梯度/数值溢出并记录（推理阶段通常禁用梯度）。

### 4.6 结构化数据 MLP 模块
- **功能说明**：对结构化临床特征进行非线性映射，捕获特征交互。
- **输入/输出**：
  - 输入：`structured_tensor`。
  - 输出：结构化特征向量 `struct_feat` (dim = d_struct_out)。
- **数据结构**：标准化后的浮点张量。
- **内部逻辑**：多层感知机（Dense + ReLU/BatchNorm/Dropout）前向。
- **异常处理**：输入维度不匹配时报错；对 NaN/Inf 进行检测并拒绝。

### 4.7 多模态融合模块
- **功能说明**：融合文本、时序和结构化特征，生成统一表征。
- **输入/输出**：
  - 输入：`text_feat`, `ts_feat`, `struct_feat`。
  - 输出：融合特征 `fusion_feat`。
- **数据结构**：多个向量张量，维度在融合前对齐。
- **内部逻辑**：
  - 拼接特征向量，输入注意力/加权层；
  - 通过可学习权重生成各模态注意力系数；
  - 融合方式：`fusion_feat = concat([a_text * text_feat, a_ts * ts_feat, a_struct * struct_feat])`。
- **异常处理**：维度不一致时抛出异常；对缺失模态给予默认零向量并调整权重。

### 4.8 风险预测与评分模块
- **功能说明**：基于融合特征输出 AKI 风险概率与等级。
- **输入/输出**：
  - 输入：`fusion_feat`。
  - 输出：`risk_score` ∈ [0,1]，`risk_category`（Low/Medium/High）。
- **数据结构**：融合向量 → 全连接层输出标量，再经 Sigmoid。
- **内部逻辑**：线性层 → Sigmoid 得到概率；依据阈值分层（如 <0.33 低，0.33-0.66 中，≥0.66 高）。
- **异常处理**：数值异常（NaN/Inf）时拒绝输出并记录；模型未加载时返回错误。

## 5. 数据流程图（Data Flow Diagram, DFD）
```
[User Upload]
     |
     v
[Web UI Form] --JSON--> [FastAPI /predict]
     |                        |
     |                        v
     |                 [数据预处理与校验]
     |                        |
     |                        v
     |            [文本编码] [时序编码] [结构化编码]
     |                        |    |      |
     |                        +----+------+---> [多模态融合]
     |                                       |
     |                                       v
     |                                 [风险预测]
     |                                       |
     +<--------------------------------------+
                [概率 & 等级输出展示]
```

## 6. 核心算法说明（Core Algorithm Description）
### 6.1 多模态融合策略
- 三个模态特征向量先按维度对齐后拼接；
- 使用可学习注意力系数 `a_text`, `a_ts`, `a_struct`（Softmax 归一化），对各模态特征加权；
- 融合向量 `fusion_feat = concat(a_text * text_feat, a_ts * ts_feat, a_struct * struct_feat)`。

### 6.2 注意力/拼接逻辑
- 计算注意力 logits：`logits = W_att @ [text_feat; ts_feat; struct_feat] + b_att`；
- `att_weights = softmax(logits)`；
- 加权后再进行线性层或小型 MLP，以提升特征交互能力。

### 6.3 概率变换
- 输出层使用 Sigmoid：`risk_score = sigmoid(W_out · fusion_feat + b_out)`，确保概率位于 [0,1]。

### 6.4 风险分层规则
- 低风险：`risk_score < 0.33`
- 中风险：`0.33 ≤ risk_score < 0.66`
- 高风险：`risk_score ≥ 0.66`
- 规则可根据校准曲线或临床反馈调整。

## 7. 关键伪代码（Pseudocode）
### 7.1 数据预处理管线
```pseudo
function preprocess(clinical_features, note_text, timeseries):
    validate_schema(clinical_features, note_text, timeseries)
    struct_tensor = encode_structured(clinical_features)
    text_tokens = tokenize(note_text, max_len)
    text_tensor = pad_or_truncate(text_tokens, max_len)
    ts_aligned = align_timeseries(timeseries, target_steps)
    ts_tensor = normalize(ts_aligned)
    return struct_tensor, text_tensor, ts_tensor
```

### 7.2 文本编码
```pseudo
function encode_text(text_tensor, attention_mask):
    if use_bert:
        outputs = bert_model(text_tensor, attention_mask)
        text_feat = outputs[CLS]
    else:
        embeddings = embed(text_tensor)
        text_feat = mean_pool(embeddings)
    return text_feat
```

### 7.3 时序编码
```pseudo
function encode_timeseries(ts_tensor):
    h0, c0 = zeros()
    outputs, (hn, cn) = lstm(ts_tensor, (h0, c0))
    ts_feat = pool(outputs, mode="mean")
    return ts_feat
```

### 7.4 多模态融合前向
```pseudo
function fuse(text_feat, ts_feat, struct_feat):
    concat_feat = concatenate([text_feat, ts_feat, struct_feat])
    logits = linear_att(concat_feat)
    att_weights = softmax(logits)  # size 3
    weighted = concatenate([
        att_weights[0] * text_feat,
        att_weights[1] * ts_feat,
        att_weights[2] * struct_feat
    ])
    fusion_feat = fusion_mlp(weighted)
    return fusion_feat
```

### 7.5 AKI 预测流程
```pseudo
function predict_AKI(clinical_features, note_text, timeseries):
    struct_tensor, text_tensor, ts_tensor = preprocess(...)
    text_feat = encode_text(text_tensor, attention_mask(text_tensor))
    ts_feat = encode_timeseries(ts_tensor)
    struct_feat = mlp_struct(struct_tensor)
    fusion_feat = fuse(text_feat, ts_feat, struct_feat)
    risk_logit = linear_out(fusion_feat)
    risk_score = sigmoid(risk_logit)
    risk_category = stratify(risk_score)
    return {"risk_score": risk_score, "risk_category": risk_category}
```

## 8. 数据库设计（示例）
- **Table: prediction_history**
  - id (PK), patient_id, request_payload (JSON), risk_score (float), risk_category (string), model_version, created_at (timestamp)
- **Table: audit_logs**
  - id (PK), level, message, detail (JSON), created_at (timestamp)
- **Table: dataset_uploads**
  - id (PK), uploader, file_name, checksum, upload_time, status, remarks

## 9. 接口设计（API Design）
### 9.1 端点定义
- `POST /predict`
  - 描述：接收多模态数据，返回 AKI 风险预测。

### 9.2 输入 Schema（JSON）
```json
{
  "clinical_features": {"age": 65, "sex": "M", "creatinine": 1.2, ...},
  "note_text": "Patient underwent CABG...",
  "timeseries": [
    {"time": "00:00", "HR": 80, "MAP": 75},
    {"time": "00:05", "HR": 82, "MAP": 78}
  ]
}
```

### 9.3 输出 Schema（JSON）
```json
{
  "risk_score": 0.71,
  "risk_category": "High",
  "model_version": "v1.0",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### 9.4 错误处理格式
```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "missing field: clinical_features.age",
  "detail": {...}
}
```

## 10. 用户界面设计说明（UI Design Specification）
- **页面布局**：
  - 顶部标题区：系统名称与版本。
  - 左侧输入区：
    - 结构化字段表单（年龄、性别、关键实验室指标等）。
    - 文本输入框（临床笔记）。
    - 时序数据上传控件（CSV/JSON）。
  - 右侧结果区：风险概率条形图/数值、风险等级标签、时间戳、模型版本。
  - 底部日志/提示区：显示输入校验结果与错误提示。
- **交互流程**：填写/上传 → 点击“预测” → 调用 API → 显示风险分数与等级 → 可导出 JSON/截图。
- **交互流程图**：
```
[输入表单] --点击预测--> [发送 /predict 请求] --成功--> [显示风险分数+等级]
                                               \--失败--> [显示错误提示]
```

## 11. 安全性设计（Security Considerations）
- 输入验证：前后端双重校验字段类型、长度、数值范围；拒绝恶意或缺失数据。
- 数据清洗：移除脚本标签、控制字符，防止 XSS/注入；日志脱敏。
- 传输安全：推荐 HTTPS；对内部调用使用鉴权 token；必要时启用 IP 白名单。
- 速率限制：为 `/predict` 设置每 IP/用户速率阈值，防止滥用。
- 敏感数据：限制存储时间，使用访问控制；数据库字段按需脱敏或加密。

## 12. 性能设计（Performance Considerations）
- 模型加载优化：应用启动时预加载 PyTorch 模型；使用半精度或 TorchScript 提升推理速度。
- 并发与异步：FastAPI 使用异步路由；批量或队列推理以提升吞吐。
- 缓存策略：对相同输入的短期重复请求可选缓存；模型权重与 tokenizer 缓存于内存/磁盘。
- 资源监控：记录响应时间与内存占用，触发告警；支持 GPU/CPU 切换配置。

## 13. 设计总结（Design Summary）
本设计说明书定义了“基于多模态异构数据的心脏外科术后急性肾损伤风险预测系统 V1.0”的目标、架构、模块、数据流程与核心算法。系统采用 Web 前端 + FastAPI 后端 + PyTorch 模型的分层结构，融合文本、时序与结构化特征以生成 AKI 风险概率与分层结果。设计兼顾安全、性能与合规要求，可支撑临床科研场景的部署与迭代。
