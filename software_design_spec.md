# 《基于多模态异构数据的心脏外科术后急性肾损伤风险预测系统 V1.0》软件设计说明书

> 版本：V1.0  
> 状态：提交版（用于中国软件著作权登记备案）  
> 作者：研发团队  
> 日期：2024-05-01

本设计说明书依据中国国家版权局软件著作权登记要求编制，涵盖设计目的、总体方案、详细模块设计、数据与接口规范、安全与性能设计等，确保文档完整、准确、可追溯。

---

## 1. 设计目的（Design Objectives）
### 1.1 软件目的
- 提供针对心脏外科术后患者的急性肾损伤（Acute Kidney Injury, AKI）风险预测功能，输出风险概率与风险等级，辅助临床决策和随访管理。
- 构建可重复、可扩展的多模态数据处理与深度学习推理管道，支持临床研究与模型迭代。
- 提供统一的 Web 端交互与 RESTful API 接口，方便不同角色（医生、科研人员、管理员）安全使用。

### 1.2 系统角色
- **临床研究人员/医生**：上传或录入患者术前、术中、术后数据，获取 AKI 风险预测结果与风险等级建议。
- **数据管理员**：管理数据集版本、日志、审计记录，监督数据合规性与质量。
- **系统运维人员**：维护后端服务与模型权重，监控性能，进行版本更新、容灾与备份。
- **审计/质控人员**：查看操作日志，核查模型使用合规性、性能与输出稳定性。

### 1.3 在临床研究流程中的位置
- 位于“术后风险评估”环节：在临床数据采集、脱敏、清洗后，模型对患者进行风险预测，为医生的干预措施（如监测、用药调整）提供量化参考。
- 可与医院科研数据仓库、电子病历（EMR）或临床试验管理系统集成，输出结构化报告或可视化结果，支持批量推理与个案分析。
- 支持模型迭代验证：通过历史推理记录与真实结局对比，开展回顾性研究或前瞻性验证。

### 1.4 约束与假设
- **合规与隐私**：假设输入数据已完成脱敏处理，满足《个人信息保护法》与医疗数据管理规范；系统部署在院内或受控网络环境。
- **运行环境**：服务器具备 Python 3.10+、PyTorch、FastAPI 运行条件，支持 GPU/CPU 推理；前端浏览器需支持现代 JavaScript 与 HTTPS。
- **数据完整性**：输入遵循预定义 schema，缺失值按预处理策略填补；文本编码可选择 BERT 或精简占位模型；时序数据需满足最小采样率与时间对齐要求。
- **性能限制**：单次预测延迟目标 < 2s（CPU 模式）/< 1s（GPU 模式），并发能力与硬件资源成正比；如需大规模批量推理，可启用异步/批处理模式。

### 1.5 质量属性与目标
- **可靠性**：接口具备输入校验与异常处理，模型推理失败率 < 0.1%。
- **可维护性**：模块化设计，代码层次清晰，支持模型、预处理、前端的独立更新。
- **可扩展性**: 支持新增模态（如影像）、新增特征字段及新的风险分层规则。
- **可审计性**：完整的请求、响应、错误日志与模型版本信息记录，便于事后追溯。

---

## 2. 总体设计（Overall System Design）
### 2.1 系统概述
系统由 Web 前端、FastAPI 后端、数据预处理与模型推理管道构成。前端负责数据录入、文件上传、结果展示；后端提供 RESTful API、统一数据校验、模型调用与日志记录；模型管道包括文本编码、时序编码、结构化数据 MLP、多模态融合与风险评分，并输出可视化与可解释信息。

### 2.2 模块分解
- **Web UI 模块**：数据输入表单、文本框、时序信号上传、进度提示、结果展示与导出。
- **Backend API 模块（FastAPI）**：路由与控制器、请求/响应模型、模型服务调用、错误处理、健康检查。
- **数据预处理与校验模块**：字段类型校验、缺失值填补、归一化/标准化、时序对齐与截断、文本清洗、异常数据剔除。
- **文本编码模块（BERT/Dummy Encoder）**：将临床笔记 note_text 编码为固定维度向量，支持 GPU/CPU 推理。
- **时序编码模块（LSTM/GRU 可切换）**：处理术中监测的时序信号（如血压、心率、尿量），输出隐藏状态向量。
- **结构化数据 MLP 模块**：对临床结构化特征进行线性层与非线性激活组合，生成表征向量。
- **多模态融合模块**：采用拼接 + 注意力权重或门控机制融合各模态特征，增强表示能力。
- **风险预测与评分模块**：全连接层 + Sigmoid 输出风险概率，按阈值或分位数生成风险等级并提供提示语。
- **日志与历史记录模块**：存储输入摘要、模型版本、输出结果、错误与审计信息；支持分页查询与导出。

### 2.3 组件职责
- **UI 组件**：
  - 收集结构化特征、自由文本、时序信号上传。
  - 显示风险概率、风险等级、时间戳、模型版本及提示语。
  - 提示输入格式错误或推理异常，支持重新提交与导出结果。
- **API 控制器**：
  - 处理 POST /predict、GET /health、GET /history 等请求。
  - 执行数据校验、调用预处理、触发模型推理、记录日志。
- **预处理器**：
  - 保证数据质量，统一数据格式，生成标准张量输入。
  - 对异常值、缺失值与时序不齐进行填补或对齐。
- **模型服务**：
  - 加载 PyTorch 模型与 tokenizer，管理 GPU/CPU 上下文。
  - 执行文本/时序/结构化编码与融合，返回概率与等级。
- **日志与存储层**：
  - 记录请求信息、响应、错误、运行耗时、模型版本。
  - 可对接数据库（如 PostgreSQL / SQLite）或文件型存储。

---

## 3. 系统架构图（System Architecture Diagram）
```
+-------------------+        +-----------------+        +--------------------+        +----------------------+        +----------------+
|     Web UI        |  -->   |   FastAPI       |  -->   |   Model Pipeline   |  -->   |   PyTorch Inference  |  -->   | Result Output  |
| (HTML/JavaScript) |        | (Backend API)   |        | (Preprocess/Fusion)|        | (Encoder + Sigmoid)  |        | (JSON/Visual)  |
+-------------------+        +-----------------+        +--------------------+        +----------------------+        +----------------+
```

---

## 4. 模块设计（Detailed Module Design）
### 4.1 Web UI 模块
#### 功能说明
- 提供结构化特征输入表单，支持数值、枚举、布尔类型。
- 提供自由文本框录入 note_text，并显示字符计数与基础格式提示。
- 支持上传 CSV/JSON 格式的时序信号文件（如时间戳-数值对），提供文件大小与格式校验。
- 调用后端 API 并显示风险概率、风险等级、模型版本、时间戳与提示语，支持结果导出（JSON/PNG）。

#### 输入/输出
- **输入**：用户填写的 clinical_features（JSON）、note_text（字符串）、timeseries 文件或 JSON。
- **输出**：后端返回的 risk_score（0~1）、risk_category、提示信息与模型版本；前端以卡片或表格形式展示。

#### 数据结构
- 前端内部使用 JavaScript 对象：
  - `clinical_features`: `{feature_name: number/string/bool}`
  - `note_text`: `string`
  - `timeseries`: `[{timestamp: ISO8601, signal_name: string, value: number}, ...]`
- 与后端交互 JSON：见第 9 节接口设计。

#### 内部逻辑
- 表单校验（必填项、数值范围、长度限制）。
- 文件选择与预览，调用前进行大小与格式校验。
- 调用 POST /predict，显示加载动画，等待响应并展示结果卡片。
- 支持“重置”、“再次预测”、“下载结果”按钮。

#### 异常处理
- 输入校验失败：前端提示具体字段错误，不发送请求。
- 后端返回 4xx/5xx：显示错误提示与重试按钮，并记录错误编号。
- 网络超时：提示用户检查网络或稍后重试。

### 4.2 Backend API 模块（FastAPI）
#### 功能说明
- 暴露预测、健康检查、历史查询等 RESTful 接口。
- 执行请求体验证、数据预处理、模型推理、结果封装与日志记录。

#### 输入/输出
- **输入**：HTTP 请求体（JSON / multipart），包含临床结构化特征、文本、时序数据或上传文件。
- **输出**：标准化 JSON 响应，包含 risk_score、risk_category、模型版本、耗时与错误码。

#### 数据结构
- 使用 Pydantic 模型定义请求/响应 schema（详见第 9 节）。

#### 内部逻辑
- 路由分发：/predict、/health、/history、/version。
- 前置处理：身份/来源校验（可选）、字段校验、异常捕获。
- 调用预处理与模型管道，组装响应并写入日志。

#### 异常处理
- Schema 校验失败：返回 400，错误详情字段列出问题项。
- 模型推理异常：返回 500，包含错误码与追踪 ID；后台记录详细堆栈。
- 不支持的媒体类型：返回 415。

### 4.3 数据预处理与校验模块
#### 功能说明
- 对 clinical_features、note_text、timeseries 进行统一清洗、标准化与校验，为模型生成张量。
- 处理缺失值、异常值、单位换算与时序对齐。

#### 输入/输出
- **输入**：原始 JSON/文件数据。
- **输出**：标准化张量或数组：`struct_tensor (N_s)`, `text_tokens (seq_len)`, `ts_tensor (T, d)`。

#### 数据结构
- 结构化特征：Python dict → pandas DataFrame / NumPy 数组。
- 文本：字符串 → 分词后 token id 列表。
- 时序：列表/表格 → (时间, 特征) 矩阵，已对齐/插值。

#### 内部逻辑
- 字段存在性与类型校验；缺失值填补（均值/中位数/零/特殊 token）。
- 数值标准化（均值-方差/最小-最大），类别特征独热编码。
- 文本清洗（去除控制字符、统一大小写、截断）。
- 时序对齐：按统一时间步长重采样/插值，截断或填充到最大长度；异常点（超物理范围）裁剪或置空。

#### 异常处理
- 发现非法值/不支持字段：记录警告，按策略剔除或置默认值。
- 无法对齐或时序长度为 0：返回 400 或使用空填充并附带警告。

### 4.4 文本编码模块（BERT 或 Dummy Encoder）
#### 功能说明
- 将 note_text 编码为固定长度向量。默认使用预训练 BERT（可裁剪层数）；在资源受限场景使用 Dummy Encoder（平均词向量或 TF-IDF + 线性投影）。

#### 输入/输出
- **输入**：token id 序列，长度 L（截断到上限）。
- **输出**：文本表征向量 `text_vec ∈ R^d_text`。

#### 数据结构
- 分词器输出：`input_ids`, `attention_mask`。
- 模型参数：BERT 权重或轻量化线性映射矩阵。

#### 内部逻辑
- 调用 tokenizer → 生成张量 → 模型前向 → 取 [CLS] 表征或平均池化输出。
- 可选择冻结 BERT，或在后端加载已微调权重。

#### 异常处理
- 过长文本：截断并记录警告。
- 编码失败：返回默认零向量并标记错误码，或终止推理。

### 4.5 时序编码模块（LSTM）
#### 功能说明
- 处理术中多通道时序信号（如血压、心率、尿量、血气指标），提取动态特征。

#### 输入/输出
- **输入**：对齐后的时序矩阵 `ts_tensor ∈ R^{T×d_ts}`。
- **输出**：隐藏状态向量 `ts_vec ∈ R^d_ts_hidden`。

#### 数据结构
- PyTorch 张量，支持 batch 维度；掩码标识有效时间步。

#### 内部逻辑
- 使用单/双向 LSTM 或 GRU；支持多层堆叠与 dropout。
- 取最后隐藏状态、平均池化或注意力池化作为输出。

#### 异常处理
- 时序全为空：返回零向量并记录警告。
- 尺寸不匹配：抛出异常，返回 400。

### 4.6 结构化数据 MLP 模块
#### 功能说明
- 对结构化临床特征进行非线性映射，提取高级表征。

#### 输入/输出
- **输入**：标准化后的向量 `struct_tensor ∈ R^{d_struct}`。
- **输出**：表征向量 `struct_vec ∈ R^{d_struct_hidden}`。

#### 数据结构
- NumPy/PyTorch 张量，含数值与独热编码列。

#### 内部逻辑
- 多层感知机：Linear → ReLU/LeakyReLU → Dropout → LayerNorm（可选）。
- 可根据特征维度动态配置隐藏层大小与深度。

#### 异常处理
- 输入存在 NaN/Inf：置零并记录警告。

### 4.7 多模态融合模块
#### 功能说明
- 将文本、时序、结构化特征融合，生成统一患者表征，提升预测准确度。

#### 输入/输出
- **输入**：`text_vec`, `ts_vec`, `struct_vec`。
- **输出**：`fusion_vec ∈ R^{d_fusion}`。

#### 数据结构
- 拼接向量与注意力权重张量。

#### 内部逻辑
- 基础策略：向量拼接 → 全连接层降维 → 非线性激活。
- 可选注意力/门控：计算各模态权重 `α_text, α_ts, α_struct`，融合为加权和；支持温度系数与归一化。
- 正则化：Dropout + LayerNorm，防止过拟合。

#### 异常处理
- 某模态缺失：使用零向量并重新归一化权重。
- 权重数值异常（NaN/Inf）：回退到均匀权重。

### 4.8 风险预测与评分模块
#### 功能说明
- 将融合向量映射到 AKI 风险概率并生成风险等级与提示。

#### 输入/输出
- **输入**：`fusion_vec`。
- **输出**：`risk_score ∈ [0,1]`，`risk_category ∈ {低, 中, 高}`，可选提示文本。

#### 数据结构
- 线性层权重、Sigmoid 函数、风险阈值配置。

#### 内部逻辑
- 前向计算：`logit = W·fusion_vec + b` → `risk_score = sigmoid(logit)`。
- 风险分层：按固定阈值（如 0.33, 0.66）或分位数动态阈值映射到等级；可配置不同临床场景下的阈值。
- 输出附带模型版本、时间戳与耗时，便于审计。

#### 异常处理
- 计算异常：返回错误码并记录日志；若概率异常（NaN/超界），回退为默认值 0.5 并标记警告。

---

## 5. 数据流程图（Data Flow Diagram, DFD）
```
[用户/医生] 
    | 上传/输入 clinical_features, note_text, timeseries
    v
[Web UI 表单校验]
    | JSON/文件
    v
[FastAPI 请求校验]
    | 合法请求
    v
[数据预处理与标准化]
    | struct_tensor, text_tokens, ts_tensor
    v
[文本编码] -> text_vec
[时序编码] -> ts_vec
[结构化编码] -> struct_vec
    | 向量
    v
[多模态融合 + 注意力]
    | fusion_vec
    v
[风险预测 Sigmoid + 分层]
    | risk_score, risk_category
    v
[结果封装/日志] -> 返回 Web UI/外部系统
```

---

## 6. 核心算法说明（Core Algorithm Description）
### 6.1 多模态融合策略
- **拼接 + 全连接**：`fusion = FC([text_vec; ts_vec; struct_vec])`，通过线性层与激活整合各模态信息。
- **注意力权重**：
  - 计算 `α_text = softmax(W_t · text_vec)`, `α_ts = softmax(W_s · ts_vec)`, `α_struct = softmax(W_c · struct_vec)`。
  - 归一化权重后加权求和：`fusion = α_text * text_vec + α_ts * ts_vec + α_struct * struct_vec`。
- **门控机制（可选）**：使用 sigmoid 门控控制各模态贡献，`g = sigmoid(W_g [text; ts; struct])`，`fusion = g ⊙ concat(...)`。

### 6.2 概率变换（Sigmoid）
- 使用 Sigmoid 将 logit 映射到 [0,1]：`risk_score = 1 / (1 + exp(-logit))`。
- 可选温度缩放：`risk_score = sigmoid(logit / T)` 以校准模型输出。

### 6.3 风险分层规则
- 固定阈值示例：
  - `risk_score < 0.33` → 低风险
  - `0.33 ≤ risk_score < 0.66` → 中风险
  - `risk_score ≥ 0.66` → 高风险
- 动态阈值示例：根据历史分布计算分位点（如 30%、70%），适配特定科室或人群。
- 可附加规则：若关键指标异常（如尿量极低、乳酸升高），在同等概率下提高一级风险。

### 6.4 置信度与提示
- 计算输入完整度、模态缺失情况、模型输出温度缩放后置信度区间（Platt scaling/温度校准）。
- 对缺失模态或明显异常的输入，在结果中附加“数据质量较低”提示。

### 6.5 解释性输出（可选）
- 特征重要性：可使用逐特征梯度、注意力权重或简单特征贡献排序，供医生参考。
- 文本注意力：显示最具权重的关键词；时序注意力：标出关键时间段。

---

## 7. 关键伪代码（Pseudocode）
### 7.1 数据预处理流水线
```pseudo
function preprocess(payload):
    validate_schema(payload)
    struct = payload.clinical_features
    text = payload.note_text
    ts = payload.timeseries

    # 结构化特征
    struct = fill_missing(struct, strategy="median")
    struct = normalize(struct, method="standard")
    struct_tensor = to_tensor(struct)

    # 文本
    text = clean_text(text)
    tokens = tokenizer(text, max_len=MAX_LEN)

    # 时序
    ts_aligned = align_timeseries(ts, step=STEP, max_len=T_MAX)
    ts_tensor = to_tensor(ts_aligned)

    return struct_tensor, tokens, ts_tensor
```

### 7.2 文本编码
```pseudo
function encode_text(tokens):
    input_ids, attention_mask = tokens
    hidden = BERT(input_ids, attention_mask)
    text_vec = pool(hidden, mode="cls")
    return text_vec
```

### 7.3 时序编码
```pseudo
function encode_timeseries(ts_tensor):
    hidden_states, _ = LSTM(ts_tensor)
    ts_vec = pool(hidden_states, mode="last")
    return ts_vec
```

### 7.4 多模态融合前向
```pseudo
function fuse(text_vec, ts_vec, struct_vec):
    concat_vec = concatenate([text_vec, ts_vec, struct_vec])
    attn_weights = softmax(W_attn * concat_vec)
    fusion_vec = attn_weights_text * text_vec + attn_weights_ts * ts_vec + attn_weights_struct * struct_vec
    fusion_vec = dropout(layer_norm(fusion_vec))
    return fusion_vec
```

### 7.5 AKI 风险预测工作流
```pseudo
function predict(payload):
    struct_tensor, tokens, ts_tensor = preprocess(payload)
    text_vec = encode_text(tokens)
    ts_vec = encode_timeseries(ts_tensor)
    struct_vec = mlp_struct(struct_tensor)

    fusion_vec = fuse(text_vec, ts_vec, struct_vec)
    logit = linear(fusion_vec)
    risk_score = sigmoid(logit)
    category = stratify(risk_score, thresholds)

    return {"risk_score": risk_score, "risk_category": category}
```

---

## 8. 数据库设计（示例）
| 表名 | 作用 | 主要字段 | 备注 |
| --- | --- | --- | --- |
| `inference_history` | 存储每次推理请求与结果 | `id (PK)`, `request_hash`, `risk_score`, `risk_category`, `model_version`, `created_at`, `status` | 便于审计与重放 |
| `upload_logs` | 上传记录与校验结果 | `id`, `file_name`, `uploader`, `size`, `checksum`, `validate_status`, `created_at` | 支持文件级追踪 |
| `system_events` | 系统事件与错误日志 | `id`, `level`, `message`, `trace_id`, `created_at` | 对接监控告警 |
| `model_versions` | 模型版本元数据 | `version`, `path`, `hash`, `created_at`, `notes` | 支持滚动更新 |

表设计示例（PostgreSQL）：
```sql
CREATE TABLE inference_history (
    id SERIAL PRIMARY KEY,
    request_hash VARCHAR(64),
    risk_score NUMERIC(5,4),
    risk_category VARCHAR(10),
    model_version VARCHAR(32),
    created_at TIMESTAMP DEFAULT NOW(),
    status VARCHAR(20),
    payload_summary JSONB
);
```

---

## 9. 接口设计（API Design）
### 9.1 端点定义
- `POST /predict`：单次风险预测。
- `POST /predict/batch`（可选）：批量预测，接受数组输入。
- `GET /health`：健康检查，返回服务状态与模型加载情况。
- `GET /history?limit=&offset=`：查询历史推理记录（若启用存储）。
- `GET /version`：返回当前模型与应用版本。

### 9.2 输入 Schema（示例，JSON）
```json
{
  "clinical_features": {
    "age": 65,
    "sex": "M",
    "creatinine_baseline": 1.0,
    "cpb_time_min": 120,
    "sofa_score": 6
  },
  "note_text": "患者术后转入ICU，尿量偏低，血压平稳...",
  "timeseries": [
    {"timestamp": "2024-01-01T10:00:00Z", "signal": "urine_output", "value": 20},
    {"timestamp": "2024-01-01T10:05:00Z", "signal": "urine_output", "value": 18}
  ]
}
```

### 9.3 输出 Schema（示例，JSON）
```json
{
  "risk_score": 0.72,
  "risk_category": "高",
  "model_version": "v1.0",
  "inference_time_ms": 850,
  "message": "建议加强肾功能监测，复查尿量与乳酸",
  "trace_id": "abc123"
}
```

### 9.4 错误返回格式
```json
{
  "error_code": "VALIDATION_ERROR",
  "error_message": "field 'age' is missing or invalid",
  "trace_id": "err-001",
  "status": 400
}
```

### 9.5 参数与类型说明
- `clinical_features`：对象，数值/字符串/布尔混合；必填字段可通过配置定义。
- `note_text`：字符串，可为空；最大长度限制（如 2048 字符）。
- `timeseries`：数组；每项包含 `timestamp`(ISO8601)、`signal`(字符串)、`value`(数值)。

### 9.6 安全与速率限制（接口层）
- 支持 Token/Key 鉴权（如 `Authorization: Bearer <token>`）。
- 可选限流策略（如 IP 级 QPS 限制、全局并发限制）。
- 跨域控制：只允许受信任域名访问。

---

## 10. 用户界面设计说明（UI Design Specification）
### 10.1 页面布局
- 顶部导航：系统名称、模型版本、帮助入口。
- 主面板分区：
  - **左侧输入区**：
    - 结构化特征表单：分组展示（基础信息、术中信息、实验室指标）。
    - 文本输入框：多行文本，显示字符计数。
    - 时序上传控件：文件拖拽或选择，支持 CSV/JSON，显示校验结果。
  - **右侧结果区**：
    - 风险概率与等级卡片，使用进度条或环形图表示概率。
    - 提示语、模型版本、推理耗时。
    - 历史记录或批量结果列表（可选）。
- 底部区域：提交按钮、重置按钮、下载/导出按钮。

### 10.2 交互流程
```
[填入结构化特征] -> [填入/粘贴文本] -> [上传时序文件] -> [点击预测]
    | 校验失败: 红色提示并聚焦到字段
    | 校验通过: 调用后端
    v
[显示加载动画] -> [展示结果卡片/表格] -> [可导出或再次预测]
```

### 10.3 输入与结果展示
- 表单字段具备占位提示与合法范围提示；必填项用星号标记。
- 结果区：
  - 风险概率（0~1）显示为百分比；
  - 风险等级（低/中/高）以颜色区分；
  - 附加提示：数据质量、关键特征影响。
- 可选：图表显示时序信号的关键段落与注意力权重高的时间窗。

---

## 11. 安全性设计（Security Considerations）
### 11.1 输入验证与数据清洗
- 后端使用 Pydantic 强制类型检查；数值范围与枚举合法性校验。
- 文本过滤控制字符，限制长度，防止注入类攻击。
- 文件上传限制大小与文件类型，计算哈希用于完整性校验。

### 11.2 数据保护
- 所有接口仅通过 HTTPS 暴露；敏感日志脱敏（如姓名、住院号）。
- 对接数据库时使用最小权限账户，启用加密存储（如磁盘加密、传输加密）。
- 不在日志中记录完整原始文本和全量时序，仅保留摘要与哈希。

### 11.3 认证与授权（可选配置）
- 支持基于 Token 的认证；可与院内单点登录（SSO）集成。
- 不同角色可限制访问批量预测与历史查询接口。

### 11.4 速率限制与防护
- 基于 IP 或用户的速率限制与并发限制，防止滥用。
- 输入大小限制与超时控制，防止大请求阻塞。

### 11.5 审计与追踪
- 为每次请求生成 trace_id，记录时间、调用方、模型版本、状态码。
- 异常与高风险结果可触发告警或邮件通知（可选）。

---

## 12. 性能设计（Performance Considerations）
### 12.1 模型加载与资源管理
- 后端启动时预加载 PyTorch 模型与 tokenizer，避免重复加载。
- 支持 GPU/CPU 自动选择；在多实例部署时使用模型权重共享存储。

### 12.2 推理优化
- 使用批量或异步推理：对短时间内的多个请求合并为小批次，减少硬件切换开销。
- 开启浮点精度优化（如 FP16）以提升 GPU 推理速度（条件允许时）。
- 对文本截断与时序对齐进行长度上限限制，控制推理复杂度。

### 12.3 缓存策略
- 对相同 `request_hash` 的重复请求可命中缓存，直接返回历史结果。
- 可缓存 tokenizer 结果与特征标准化参数，减少重复计算。

### 12.4 可用性与容错
- 健康检查端点 + 负载均衡，支持多实例部署。
- 异常自动重试（有限次）或降级到 Dummy Encoder/CPU 模式。

---

## 13. 设计总结（Design Summary）
本说明书系统性描述了《基于多模态异构数据的心脏外科术后急性肾损伤风险预测系统 V1.0》的设计目标、架构、模块、数据流、核心算法、接口、UI、安全与性能方案。通过多模态融合（结构化、文本、时序）与可配置的风险分层规则，系统可为心脏外科术后患者提供可靠的 AKI 风险评估。模块化设计保障了可维护性与可扩展性，配合日志、审计与安全策略，满足临床研究与合规要求，为后续模型迭代与产品化落地奠定基础。

---

## 14. 补充设计细节与合规说明
### 14.1 合规要点
- 明确数据脱敏责任：数据提供方需在输入前完成脱敏，并出具合规声明。
- 日志留痕：推理请求与响应的摘要存储期限建议不低于 12 个月，供审计与科研复核。
- 数据跨境限制：禁止将患者数据上传至境外服务器；如需多院区协作，采用联邦学习或参数交换。

### 14.2 开发与交付工艺
- 代码管理：使用 Git 分支策略（main/dev/feature），提交信息需包含任务编号。
- 代码审查：重要模块（预处理、模型推理）须经双人审查；安全相关修改需额外审计。
- 文档同步：接口变更需同步更新设计说明书与 API 文档。

### 14.3 数据质量保障
- 设定必填字段列表（如年龄、性别、术式、基础肌酐、体外循环时间）。
- 异常值规则：给出各关键指标的合理范围（如收缩压 70–200 mmHg），超出范围标记为异常并裁剪。
- 缺失值影响评估：记录缺失率与填补方式，生成质量评分，附在推理结果中。

---

## 15. 部署与运行方案
### 15.1 部署架构
- 单机部署：FastAPI + Uvicorn/Gunicorn，加载 PyTorch 模型；适合小规模试用。
- 集群部署：Nginx 反向代理 + 多实例 FastAPI，模型文件放置共享存储或镜像打包；使用负载均衡分发请求。
- GPU 部署：在具备 CUDA 的节点预加载模型，限制显存占用；可通过环境变量控制是否启用 GPU。

### 15.2 环境要求
- 操作系统：Linux（Ubuntu/CentOS）推荐；需安装 Python 3.10+、PyTorch、FastAPI、Uvicorn。
- 依赖：`pydantic`, `numpy`, `pandas`, `torch`, `transformers`, `uvicorn`, `aiofiles` 等。
- 端口：默认 8000，可通过环境变量 `PORT` 配置；HTTPS 由外层代理实现。

### 15.3 部署步骤示例
1. 准备 Python 虚拟环境，安装依赖。  
2. 下载或挂载模型权重与 tokenizer 文件。  
3. 配置环境变量（模型路径、日志级别、GPU 开关、阈值配置）。  
4. 运行 `uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2`。  
5. 配置 Nginx 反代与 HTTPS 证书，校验健康检查端点。  
6. 启动日志收集与监控（如 Prometheus + Grafana 或 ELK）。

### 15.4 运行时监控指标
- QPS、平均/90%/99% 延迟、并发连接数。
- 模型推理耗时拆解（预处理、各模态编码、融合、Sigmoid）。
- 失败率与错误码分布；GPU 利用率与显存占用（如适用）。
- 日志样本：每分钟抽样记录结构化摘要，便于观测波动。

---

## 16. 运维与监控设计
### 16.1 健康检查与自愈
- `/health` 返回应用状态、模型加载状态、依赖可用性；异常时触发容器重启或告警。
- 周期性自检：尝试空载推理并记录耗时，发现显著退化时提示运维。

### 16.2 日志管理
- 分级日志：INFO 记录关键流程，WARN 记录异常输入，ERROR 记录推理失败与堆栈。
- 日志格式：JSON 行格式，字段包含 timestamp、trace_id、route、status、latency_ms、model_version。
- 日志轮转：按文件大小或天数轮转，保留周期满足院内规定。

### 16.3 告警策略
- 失败率、延迟、异常输入比例、GPU 温度/利用率超过阈值时告警。
- 高风险结果数量突增时可触发通知，辅助临床关注。

---

## 17. 备份与恢复
### 17.1 备份内容
- 模型权重、tokenizer、配置文件与阈值表。
- 历史推理记录与系统日志（脱敏后）。
- 部署脚本与环境配置（requirements、Dockerfile 等）。

### 17.2 备份策略
- 定期全量备份（每日/每周）+ 增量备份（按小时或事件触发）。
- 异地备份或多副本存储；校验备份完整性（哈希对比）。

### 17.3 恢复流程
1. 从备份仓库恢复模型与配置文件。  
2. 恢复数据库或日志文件，验证版本一致性。  
3. 重启服务并执行健康检查与回归测试。  
4. 如有模型版本差异，记录变更并更新说明书。

---

## 18. 测试计划
### 18.1 测试类型
- **单元测试**：预处理、文本编码、时序对齐、阈值分层逻辑。
- **集成测试**：端到端接口调用，校验输入校验、预处理、模型推理、响应封装。
- **性能测试**：并发压测、批量推理延迟、内存/显存占用。 
- **安全测试**：鉴权校验、输入攻击（超长文本、异常文件）、越权访问模拟。
- **回归测试**：模型或预处理更新后的基准对比，确保输出稳定。

### 18.2 测试用例示例
- 缺失必要字段时返回 400，错误消息包含缺失字段名。
- 超长文本自动截断，风险结果正常返回，并附加警告。
- 时序全为空时返回零向量，风险概率仍可计算且标记数据质量低。
- GPU 不可用时自动切换 CPU，延迟上升但结果一致。
- 批量预测输入 10 条记录，平均延迟低于设定阈值。

### 18.3 验收标准
- 功能正确率 ≥ 99%；关键路径无阻塞错误。
- 预测延迟满足性能指标；日志与审计记录完整。
- 安全与合规检查通过（身份、权限、输入防护）。

---

## 19. 数据字典与字段说明（示例）
| 字段 | 类型 | 单位/取值 | 说明 | 预处理规则 |
| --- | --- | --- | --- | --- |
| age | int | 年 | 患者年龄 | 必填，范围 0–120，超界裁剪 |
| sex | string | M/F | 性别 | 枚举校验 |
| creatinine_baseline | float | mg/dL | 基线肌酐 | 缺失填补中位数，范围 0.1–15 |
| cpb_time_min | float | 分钟 | 体外循环时间 | 负值无效，缺失填补均值 |
| sofa_score | int | 分 | SOFA 评分 | 范围 0–24，超界裁剪 |
| note_text | string | - | 临床文本 | 最大长度 2048，控制字符移除 |
| timeseries.timestamp | string | ISO8601 | 时间戳 | 校验格式，无法解析则丢弃记录 |
| timeseries.signal | string | - | 信号名称 | 受控词表校验（如尿量、血压） |
| timeseries.value | float | - | 信号值 | 非数值置空；异常范围裁剪 |

---

## 20. 配置项清单（示例）
| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_PATH` | `models/aki_model.pt` | PyTorch 模型路径 |
| `TOKENIZER_PATH` | `models/tokenizer/` | 文本 tokenizer 路径 |
| `USE_GPU` | `false` | 是否启用 GPU 推理 |
| `BATCH_SIZE` | `1` | 推理批量大小 |
| `TEXT_MAX_LEN` | `256` | 文本最大 token 长度 |
| `TS_MAX_LEN` | `512` | 时序最大步数 |
| `THRESHOLDS` | `[0.33, 0.66]` | 风险分层阈值 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `RATE_LIMIT_QPS` | `100` | 每 IP 每秒请求上限（示例） |
| `CACHE_TTL_SEC` | `3600` | 结果缓存有效期 |

---

## 21. 版本控制与发布策略
- 采用语义化版本号（MAJOR.MINOR.PATCH），模型或接口变更提升 MINOR；兼容性破坏提升 MAJOR。
- 发布流程：开发 → 代码审查 → 预生产环境验证 → 生产发布；每次发布记录发布单与回滚预案。
- 回滚策略：保留上一稳定版本镜像与模型；发布失败时切换回稳定版本并恢复配置。

---

## 22. 变更记录（示例）
| 版本 | 日期 | 变更内容 | 责任人 |
| --- | --- | --- | --- |
| 1.0.0 | 2024-05-01 | 初始版本，完成多模态推理与前后端接口 | 团队 |
| 1.1.0 | 2024-06-15 | 新增批量预测接口与缓存策略 | 团队 |
| 1.2.0 | 2024-07-20 | 优化时序注意力、增加数据质量评分 | 团队 |

---

## 23. 术语表
- **AKI**：Acute Kidney Injury，急性肾损伤。
- **CPB**：Cardiopulmonary Bypass，体外循环。
- **Sigmoid**：常用激活函数，将实数映射到 (0,1)。
- **Attention**：注意力机制，用于计算不同模态或时间步的重要性。
- **Tokenizer**：文本分词器，将文本转为模型可接受的 token 序列。
- **Trace ID**：请求追踪标识，用于日志关联与排错。
- **LayerNorm/Dropout**：深度学习常用正则化与归一化技术。

---

## 24. 附录：示例序列图与流程细化
### 24.1 接口调用序列图（ASCII）
```
用户 → Web UI: 填写表单/上传文件
Web UI → FastAPI: POST /predict (JSON/文件)
FastAPI → Validator: 校验 schema/字段
Validator → Preprocessor: 结构化/文本/时序预处理
Preprocessor → Model Service: 规范化张量输入
Model Service → Text Encoder: encode_text()
Model Service → TS Encoder: encode_timeseries()
Model Service → MLP: encode_struct()
Model Service → Fusion: fuse()
Model Service → Scoring: sigmoid + stratify
Model Service → FastAPI: 返回结果
FastAPI → Logger: 写入历史/审计
FastAPI → Web UI: 响应 risk_score/category
Web UI → 用户: 展示结果与提示
```

### 24.2 端到端异常分支示例
- **分支 A：必填字段缺失**  
  - FastAPI 校验失败 → 400 + 错误字段列表 → UI 高亮并提示。  
- **分支 B：文件格式不符**  
  - 预处理发现 CSV 缺少必须列 → 返回 400 + 说明；记录校验失败日志。  
- **分支 C：模型加载失败**  
  - /health 返回异常状态，告警并触发自动重启；预测接口短暂不可用。  
- **分支 D：GPU 不可用**  
  - 回退到 CPU 推理，增加耗时；响应包含 `device=fallback_cpu` 提示。  
- **分支 E：缓存命中**  
  - 若相同请求重复提交，直接返回历史结果，延迟显著降低。  

### 24.3 性能基线与容量估算
- 假设单实例 CPU 推理平均 800 ms，QPS 目标 2；多实例线性扩展。
- GPU 推理平均 300 ms，适合高并发；批量模式下吞吐量可提升 2–4 倍。
- 预处理耗时占 20–40%，可通过并行或缓存 tokenizer 结果优化。

### 24.4 可扩展模态预留
- 影像模态（如超声/CT）：预留 `image_vec` 接口，统一在融合模块拼接。
- 实验室数据流：新增时间序列/事件表，复用时序编码器。
- 规则引擎：在评分后可加入规则校正层，对特定高危特征调整分层。

---

## 25. 结论与提交声明
本设计说明书篇幅扩展且覆盖了中国软件著作权登记所需的完整要素，包括设计目的、架构、模块、数据与接口规范、安全、性能、部署、测试、合规与变更记录等。文档中的算法、流程与接口细节可直接指导实现与验收，也可作为申报材料的正式附件提交。

---

## 26. 数据示例与校验规则扩展
### 26.1 结构化数据示例
```json
{
  "age": 58,
  "sex": "F",
  "height_cm": 160,
  "weight_kg": 60,
  "bmi": 23.4,
  "hypertension": true,
  "diabetes": false,
  "creatinine_baseline": 0.9,
  "cpb_time_min": 105,
  "icu_los_hours": 12
}
```

### 26.2 文本数据示例
- “患者术后尿量较前下降，血压维持 110/70，乳酸 2.0，血红蛋白 110，建议密切监测尿量与肾功能。”
- 文本校验：
  - 最大长度 2048；
  - 去除 HTML/JS 片段与控制字符；
  - 统一全角/半角符号。

### 26.3 时序数据示例（CSV）
```
timestamp,signal,value
2024-01-01T10:00:00Z,urine_output,25
2024-01-01T10:05:00Z,urine_output,22
2024-01-01T10:00:00Z,map,75
2024-01-01T10:05:00Z,map,78
```

### 26.4 校验规则补充
- 时间戳必须单调不减，同一信号的时间差不低于 1 分钟（示例值）。
- 同一时间戳重复记录以最后一条为准，或对数值取平均。
- 允许的信号列表需配置在后端，未识别信号将被忽略并产生警告。

---

## 27. 模型训练概述（供备案参考）
### 27.1 数据集构成
- 多中心回顾性数据，涵盖术前、术中、术后指标；
- 文本来源：术后病程记录与护理记录；
- 时序信号：尿量、动脉压、中心静脉压、心率、血氧等。

### 27.2 训练流程摘要
1. 数据脱敏与分层抽样，划分训练/验证/测试集。
2. 预处理与特征工程：数值标准化、类别独热、文本分词、时序重采样。
3. 模型结构：文本编码（BERT-base 或蒸馏版本）、时序编码（双向 LSTM）、结构化 MLP，多模态融合后接二分类头。
4. 损失函数：二元交叉熵；优化器 AdamW；学习率调度器线性 warmup。
5. 评估指标：AUROC、AUPRC、敏感度、特异度、校准曲线；选择最佳阈值。
6. 模型固化与版本标签：保存权重、tokenizer、配置与阈值，记录 Git 提交与训练数据版本。

### 27.3 推理配置与差异
- 推理时关闭 dropout，开启 eval 模式；
- 采用温度校准参数（如 T=1.2）提升概率可靠性；
- 文本/时序长度上限与训练保持一致，避免分布漂移。

---

## 28. 风险提示模板（示例）
- **低风险**：
  - “预测结果显示 AKI 风险较低，请继续常规监测。”
- **中风险**：
  - “存在一定 AKI 风险，建议密切观察尿量、肌酐变化，必要时调整液体管理。”
- **高风险**：
  - “AKI 风险较高，请强化监测与防护措施，评估肾脏保护策略。”
- **数据质量低**：
  - “输入数据缺失或异常较多，结果置信度下降，建议补充关键指标后复核。”

---

## 29. 合规检查清单（提交前核对）
- [ ] 数据已脱敏且获得使用许可；
- [ ] 部署环境位于院内或受控网络，开启 HTTPS；
- [ ] 接口鉴权已配置（Token/白名单等）；
- [ ] 日志中无患者直接身份信息，存储周期符合院规；
- [ ] 模型版本与训练数据版本有记录，可追溯；
- [ ] 文档与实际部署配置一致，阈值与接口参数已核对；
- [ ] 备份与灾备方案已验证；
- [ ] 完成性能与安全测试，记录在案。

---

## 30. 未来规划与扩展方向
- 引入影像模态（如超声图像）并训练跨模态 Transformer，以提升早期预测能力。
- 增加可解释性模块，输出 SHAP 值或注意力可视化，辅助医生理解模型决策。
- 支持在线学习/增量更新，结合新数据定期微调或蒸馏，保持模型新鲜度。
- 与医院 HIS/EMR 深度集成，自动拉取数据并回写风险评估结果。
- 提供移动端或小程序界面，方便床旁快速录入与查询。

---

## 31. 参考文献与标准（示例）
1. KDIGO Clinical Practice Guideline for Acute Kidney Injury.
2. 中国国家版权局《计算机软件著作权登记申请表》填写规范。
3. FastAPI 官方文档与 Pydantic 模型定义规范。
4. PyTorch 官方文档（模型加载与推理最佳实践）。
5. 数据安全与隐私保护相关法律法规（《个人信息保护法》《数据安全法》）。

---

## 32. 代码与配置片段示例
### 32.1 FastAPI 路由（示例）
```python
from fastapi import FastAPI
from schemas import PredictRequest, PredictResponse
from service import inference

app = FastAPI()

@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    return await inference(payload)
```

### 32.2 模型加载（示例）
```python
import torch
from models import MultiModalModel

model = MultiModalModel()
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()
```

### 32.3 配置示例（YAML）
```yaml
model_path: models/aki_model.pt
tokenizer_path: models/tokenizer/
use_gpu: false
text_max_len: 256
ts_max_len: 512
thresholds: [0.33, 0.66]
log_level: INFO
```

---

## 33. 数据流细化（逐步说明）
1. **输入采集**：用户在 UI 填写表单/上传文件，前端初步校验。  
2. **请求封装**：前端将结构化、文本、时序数据封装为统一 JSON（或 multipart）。  
3. **API 校验**：FastAPI 使用 Pydantic 校验类型与必填字段，拒绝非法请求。  
4. **预处理**：
   - 数值标准化/缺失填补；
   - 文本分词与截断；
   - 时序重采样、插值、掩码。
5. **编码阶段**：
   - 文本编码输出 text_vec；
   - 时序编码输出 ts_vec；
   - 结构化编码输出 struct_vec。
6. **融合阶段**：拼接或注意力融合，生成 fusion_vec；正则化处理。
7. **评分阶段**：线性变换 + Sigmoid 输出 risk_score；按阈值生成风险等级。
8. **结果封装**：附加提示、模型版本、耗时、trace_id；写日志并响应。

---

## 34. 质量控制指标
| 指标 | 目标值 | 说明 |
| --- | --- | --- |
| 输入校验通过率 | ≥ 95% | 预期大部分请求符合 schema |
| 推理成功率 | ≥ 99.9% | 不包含客户端输入错误 |
| 平均延迟（CPU） | < 2s | 单条请求，含预处理 |
| 平均延迟（GPU） | < 1s | 单条请求，含预处理 |
| AUROC | ≥ 0.80 | 基于验证集 | 
| 日志完整率 | 100% | 请求均有 trace_id 记录 |

---

## 35. 角色与权限（可选实现）
- **admin**：可配置阈值、查看全部历史、管理模型版本。 
- **researcher**：可提交预测、查看自身提交的历史、导出结果。 
- **auditor**：只读访问日志与结果摘要，不可修改配置。 
- 权限控制可通过中间件实现，根据 Token 中的角色声明判断权限。

---

## 36. 国际化与本地化（可选）
- 支持中英文界面切换，文案通过 i18n 资源文件管理。
- 时间、数字、日期格式按地区配置；日志使用 UTC 保存。

---

## 37. 容器化与交付
- 提供 Dockerfile，包含基础镜像（python:3.10-slim）、依赖安装、模型复制、非 root 运行。
- 使用多阶段构建减少镜像体积；运行阶段关闭编译工具链提升安全性。
- 支持 Helm Chart 或 docker-compose 部署，参数化端口、资源限制、环境变量。

---

## 38. 批量预测与异步队列（可选）
- 增设 `POST /predict/batch` 接口，输入数组，返回数组结果；支持任务 ID 异步查询。 
- 可接入消息队列（RabbitMQ/Kafka）处理大批量请求，后台工作进程拉取队列执行推理。
- 异步结果存储在数据库或对象存储，前端通过任务 ID 轮询或回调。

---

## 39. 资源与成本优化
- 对大模型（BERT）可采用蒸馏或量化（INT8/FP16），降低延迟与资源占用。
- 通过批量推理与连接复用降低 CPU 上下文切换；对热点请求启用缓存。
- 按需扩缩容：结合监控指标自动调整实例数，避免资源浪费。

---

## 40. 可靠性与容错场景
- **模型文件缺失/损坏**：启动时校验哈希，失败则拒绝服务并告警。
- **依赖服务不可用**：健康检查失败时将实例摘除负载均衡；启用重试或降级。
- **输入爆发增长**：限流与队列削峰，保护核心推理服务；必要时返回 429。
- **磁盘/存储不足**：监控并预警；日志轮转与清理缓存释放空间。

---

## 41. 兼容性与可移植性
- 支持 CPU/GPU 切换；在无 GPU 环境下自动回退。
- 可在不同 Linux 发行版部署，依赖版本锁定以减少差异。
- 提供 requirements.txt/poetry.lock，确保环境可复现。

---

## 42. 文档与培训
- 使用在线文档或 Wiki 记录接口、配置、常见问题。
- 为临床用户提供操作手册与示例视频；为运维提供排障指南。
- 定期更新 FAQ，收集反馈并迭代。

---

## 43. 结语
本附加内容进一步细化了数据示例、训练背景、合规清单、未来扩展、容器化、批量推理、可靠性场景等信息，确保设计说明书在长度与深度上满足提交要求，并为实施、审计与运营提供充分指引。
