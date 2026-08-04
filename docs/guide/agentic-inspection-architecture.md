---
title: Agentic 原生边缘巡检架构
description: 基于 CosmoEdge 的边缘巡检 Agent 控制面方案，覆盖 Spec、编译、审批、风险评估、云边协同与落地边界。
---

# Agentic 原生边缘巡检架构

本文档记录基于 CosmoEdge 演进边缘巡检产品的第一版方案。目标不是重写现有视频分析引擎，而是在保留 CosmoEdge 实时执行面的前提下，引入一个 agentic 原生控制面，让“自然语言意图 -> 结构化 Spec -> 可审批部署计划 -> 确定性执行 -> 风险闭环”成为主路径。

## 1. 结论先行

第一版采用“双平面”设计：

- CosmoEdge 保留为确定性实时执行面，继续负责视频接入、解码、检测、跟踪、规则判断、事件输出和实时展示。
- 新增一个独立 Python 进程 `inspection-agent` 作为控制面，负责意图理解、Spec 管理、部署编译、审批治理、风险评估、报告生成和云边协同。

这意味着：

- 不重写现有 Flow Runtime。
- 不迁移到 GStreamer 编排框架。
- 不让大模型直接改动引擎内部状态。
- Web 页面从“唯一配置入口”降级为“审批、接管、报表和人工干预入口”。

## 2. 设计目标

该方案服务于以下目标：

1. 面向单机边缘盒子的一体化巡检产品。
2. 同时支持按需配置和风险评估。
3. 支持边云协同，但边缘侧可独立完成基础视频分析。
4. 采用分级自治，所有高风险变更必须经过审批。
5. 以结构化风险案件而不是自由文本作为核心输出。
6. 以自然语言和版本化 Spec 作为上层交互方式。
7. 兼容现有 CosmoEdge HTTP、MQTT、WebSocket 告警链路。
8. 适配低算力边缘设备：本地跑小模型，云端跑 VLM。

## 3. 为什么不重写 Flow 编排

CosmoEdge 当前已经有完整的业务编排与运行时：

- 实时视频链路已经打通。
- 规则节点、告警节点、可视化节点和事件链路已经闭环。
- 现有 JSON Flow 已经沉淀了业务语义，而不是单纯媒体拓扑。

如果直接重写成新的 agent workflow 或迁移到 GStreamer 编排，有三个问题：

1. GStreamer 更适合媒体数据面，不适合直接承载“告警规则、区域语义、审批治理、风险案件”这类业务控制语义。
2. 重写会把确定性实时链路和非确定性智能控制面混在一起，风险大，验证周期长。
3. CosmoEdge 现有资产已经足够支撑第一版巡检产品，重写收益低于成本。

因此第一版的原则是：

- 保留现有数据面。
- 在外部增加 agent 控制面。
- 通过确定性编译把高层 Spec 落到现有 Flow JSON 和任务配置。

## 4. 总体架构

```text
+---------------------------+        +----------------------------------+
| User / Operator           |        | Cloud VLM / OpenAI-compatible API|
| NL intent / review / case |        | Risk assessment on JPEG evidence |
+-------------+-------------+        +----------------+-----------------+
              |                                       ^
              v                                       |
+---------------------------------------------------------------+
| inspection-agent (Python sidecar)                             |
| intent -> Spec -> Plan -> approval -> apply -> assess/report  |
| spec store | policy gate | compiler | case store | scheduler   |
+----------------------+------------------------------+----------+
                       | HTTP API / WS / polling
                       v
+---------------------------------------------------------------+
| CosmoEdge                                                    |
| video ingest | decode | infer | track | rules | event | UI   |
| existing HTTP/MQTT/WebSocket outputs remain unchanged        |
+---------------------------------------------------------------+
```

分层定义：

- 数据面：CosmoEdge。
- 控制面：`inspection-agent`。
- 云评估面：OpenAI-compatible VLM 服务。
- 人工治理面：现有 Web 控制台加新增 agent 相关页面。

## 5. 角色分工

### 5.1 CosmoEdge 负责什么

- 摄像头、视频流和实时任务运行。
- 现有算法节点编排和执行。
- 事件生成、图片证据、实时推送。
- 既有 HTTP、MQTT、WebSocket、数据库落盘。
- 人工查看实时画面和历史事件。

### 5.2 inspection-agent 负责什么

- 接收自然语言意图。
- 生成和维护版本化 `InspectionSpec`。
- 将 `InspectionSpec` 编译为可执行 `DeploymentPlan`。
- 根据策略判断哪些动作可自动执行，哪些必须审批。
- 订阅告警事件，并做补偿拉取与去重。
- 触发定时巡检抓拍与云端风险复核。
- 生成 `RiskCase`、巡检报告和审计记录。

### 5.3 Web 页面负责什么

- 展示 Spec、部署计划、审批状态、案件、报告。
- 在需要时让人工接管、驳回、重试或补录说明。
- 保留现有视频配置和运行监控能力。

## 6. 第一版边界

第一版明确限制范围，避免把系统做成泛化 Agent 平台：

- 仅支持单机盒子部署，但所有核心对象保留 `deviceId` 字段。
- 本地仅跑轻量检测/分类模型。
- 云端仅接收精选 JPEG 证据，不上传连续视频流。
- 原有告警链路保持独立，云端评估失败不能阻断原始事件输出。
- 不做物理世界自动控制。
- 不做未经审批的高风险配置变更。
- 不做第二套用户认证系统，复用现有登录态与权限边界。

## 7. 核心交互模型

### 7.1 顶层流程

```text
自然语言意图
  -> InspectionSpec 草案
  -> 版本化 InspectionSpec
  -> DeploymentPlan
  -> 审批 / 自动放行
  -> 编译并下发 CosmoEdge
  -> 运行中事件 / 定时巡检
  -> 风险评估
  -> RiskCase
  -> 人工处置 / 报告归档
```

### 7.2 关键原则

- 模型只生成业务层 Spec，不直接生成底层原始 Flow。
- Spec 到 Flow 的转换由确定性编译器完成。
- 所有变更必须可 diff、可回滚、可审计。
- 所有云端判断必须有证据、有结构化结果、有失败降级路径。

## 8. 核心数据模型

### 8.1 InspectionSpec

`InspectionSpec` 是上层业务配置的唯一真相源，建议至少包含：

```json
{
  "id": "spec_xxx",
  "version": 3,
  "name": "夜间厂区巡检",
  "deviceId": "edge-box-01",
  "cameras": [],
  "objectives": [],
  "zones": [],
  "schedule": {},
  "evidencePolicy": {},
  "riskPolicy": {},
  "responsePolicy": {},
  "cloudPolicy": {}
}
```

字段语义：

- `cameras`：摄像头范围、取流方式、启停条件。
- `objectives`：巡检目标，例如 PPE、门状态、烟火、积水、异物。
- `zones`：区域、多边形、禁入区、关注区、点位。
- `schedule`：定时巡检周期、静默期、补采策略。
- `evidencePolicy`：抓拍频率、保留时长、上传规则。
- `riskPolicy`：高/中/低风险判定和升级条件。
- `responsePolicy`：自动动作、人工审批和通知策略。
- `cloudPolicy`：哪些场景可上传图片、调用哪个 VLM、超时阈值。

### 8.2 DeploymentPlan

`DeploymentPlan` 是部署前的中间产物：

```json
{
  "id": "plan_xxx",
  "specId": "spec_xxx",
  "specVersion": 3,
  "diff": [],
  "engineMutations": [],
  "preconditions": [],
  "riskLevel": "medium",
  "approvalRequired": true,
  "rollbackRef": "desired_state_xxx",
  "idempotencyKey": "apply_xxx"
}
```

它必须回答几个问题：

- 改什么。
- 改到哪里。
- 风险多高。
- 是否需要审批。
- 失败如何回滚。
- 重试是否幂等。

### 8.3 EvidenceBundle

证据对象用于绑定巡检事件与云评估输入：

```json
{
  "id": "evi_xxx",
  "cameraId": "cam_01",
  "capturedAt": "2026-07-31T10:00:00+08:00",
  "imagePath": "/data/evidence/xxx.jpg",
  "hash": "sha256:...",
  "reason": "alarm|schedule|manual",
  "relatedEventId": "evt_xxx"
}
```

### 8.4 RiskCase

`RiskCase` 是风险治理的核心对象：

```json
{
  "id": "case_xxx",
  "severity": "high",
  "status": "open",
  "confidence": 0.83,
  "evidenceRefs": ["evi_xxx"],
  "shortRationale": "未佩戴安全帽且进入高风险区域",
  "recommendedAction": "人工复核并现场确认",
  "model": "vlm-xxx",
  "humanFeedback": null
}
```

要求：

- 有结构化严重度。
- 有置信度。
- 有证据引用。
- 有短理由，不保留 chain-of-thought。
- 能接收人工反馈闭环。

### 8.5 InspectionReport

报告面向班次、日、周或专项任务汇总，建议包含：

- 巡检覆盖情况。
- 告警和风险案件统计。
- 已闭环和未闭环案件。
- 典型证据图。
- 云端评估调用统计与失败率。
- 人工处理时长与审批耗时。

## 9. Agent 对外接口

建议新增独立命名空间：

- `POST /agent/v1/intents`
- `POST /agent/v1/specs/{id}/plan`
- `POST /agent/v1/plans/{id}/approve`
- `POST /agent/v1/plans/{id}/reject`
- `POST /agent/v1/plans/{id}/apply`
- `GET /agent/v1/cases`
- `POST /agent/v1/cases/{id}/decision`
- `GET /agent/v1/reports`

接口职责：

- `intents`：自然语言转 Spec 草案。
- `plan`：从版本化 Spec 生成部署计划。
- `approve/reject`：审批流。
- `apply`：执行确定性变更。
- `cases`：案件查询与处置。
- `reports`：巡检报告查询。

## 10. 与 CosmoEdge 的集成方式

### 10.1 控制通道

`inspection-agent` 不进入帧处理主循环，只通过现有接口控制 CosmoEdge：

- 调用 `/gtw/cwai` 现有任务与布局相关 API。
- 通过现有保存接口落任务和编排。
- 依赖现有引擎负责运行态切换。

这比把 Python Agent 嵌进 C++ 视频链路更稳，也更容易回滚。

### 10.2 事件通道

事件接入采用“双路”：

1. 实时通道：监听 WebSocket 推送事件。
2. 对账通道：定期通过分页或事件查询接口补拉，防止漏读。

去重主键优先使用 `messageId` 或等价业务唯一键。

### 10.3 输出通道

原有告警输出继续保留：

- HTTP webhook
- MQTT
- WebSocket

Agent 只是并行观察者，不替换现有告警总线。

## 11. 编译策略

上层使用 `InspectionSpec`，底层仍落到现有任务和 Flow JSON。编译器必须是确定性的。

建议过程：

1. 解析 `InspectionSpec`。
2. 按目标类型匹配已有模板任务。
3. 生成或修改 Flow JSON。
4. 生成任务参数、区域、阈值、抓拍与联动配置。
5. 产出 `DeploymentPlan diff`。
6. 通过审批后调用现有 API 落地。

编译器的职责是“把业务语义映射到已有引擎能力”，不是再发明一套运行时。

## 12. 自治与审批策略

### 12.1 可自动执行

以下动作可默认自动执行：

- 信息查询。
- 证据抓拍。
- 定时巡检触发。
- 风险评估。
- 报告生成。
- Spec 草案生成。
- 已审批目标状态的自动恢复。

### 12.2 必须审批

以下动作必须人工审批：

- 启停任务。
- 修改摄像头绑定。
- 更换模型。
- 修改 Flow 结构。
- 修改区域、阈值和联动规则。
- 删除任务或配置。
- 任何对外部系统的高风险写操作。

原则很简单：读和分析可自动，改运行面必须可控。

## 13. 风险评估链路

### 13.1 触发来源

风险评估来自两类触发：

- 告警触发：已有事件出现后做二次复核。
- 定时巡检：按计划抓拍并生成“正常/异常”记录。

### 13.2 云边分工

- 本地负责检测、跟踪、区域与基础规则。
- 云端负责复杂语义判断和风险摘要。

第一版只上传经过策略筛选的 JPEG：

- 不默认上传连续视频。
- 不上传无关帧。
- 每次上传都要记录时间、哈希、模型、案件 ID。

### 13.3 失败降级

云评估可能失败，必须显式降级：

- 超时。
- 接口失败。
- 返回结构不合法。
- 模型结论低置信度。

以上情况统一标记为 `unassessed`，不能当作“无风险”。

同时：

- 原始 CosmoEdge 告警照常发送。
- Agent 可生成待人工复核案件。

## 14. 存储建议

`inspection-agent` 自带独立 SQLite 即可满足第一版：

- `specs`
- `deployment_plans`
- `approvals`
- `cases`
- `reports`
- `tool_calls`
- `audit_logs`
- `desired_state`

这样做的原因：

- 不污染现有 CosmoEdge 数据模型。
- Agent 状态可以单独迁移和回滚。
- 单机盒子部署简单。

后续若演进到多设备或云集中式控制，再考虑抽离数据库。

## 15. 认证与审计

第一版不新建第二套账户体系，直接复用现有登录态。

最小补充建议：

- 增加一个“当前会话身份信息”接口，验证 `mtk` 或现有会话令牌。
- 在 Agent 审批记录中落操作者身份、时间、动作和理由。
- 所有云端调用记录证据哈希、目标模型和返回结构摘要。

## 16. 部署形态

第一版部署形态：

- `cosmo-engine` 继续主进程运行。
- 新增 `inspection-agent` Python sidecar。
- `nginx` 将 `/agent/` 路由反代到 sidecar。

这样改动最小：

- 不破坏现有前后端部署结构。
- 不要求立刻重做鉴权体系。
- 方便单机运维和日志排查。

## 17. 实施顺序

建议按以下顺序落地：

1. 定义 `InspectionSpec`、`DeploymentPlan`、`RiskCase`、`InspectionReport` 的 schema。
2. 做一个确定性编译器，把 Spec 编译成现有 Flow JSON 与任务参数。
3. 做只读集成：先接 WebSocket 和事件补拉，不改引擎。
4. 做审批流和 `apply` 幂等落地。
5. 做定时巡检抓拍与证据归档。
6. 接入 OpenAI-compatible VLM，跑云端复核。
7. 增加案件页面、审批页面和报告页面。
8. 先 shadow mode 跑一段时间，再启用正式部署。

## 18. 测试重点

第一版不需要大而全测试矩阵，但有几类测试不能省：

- 编译器 golden tests：同一个 Spec 必须稳定产出相同结果。
- 幂等 apply 测试：重复执行不会产生脏状态。
- 审批策略测试：高风险动作不能绕过审批。
- 事件去重与补偿测试：WebSocket 丢消息后仍能补齐。
- 云评估失败测试：失败必须落 `unassessed`，不能误判通过。
- 端到端测试：自然语言 -> Spec -> 审批 -> 部署 -> 告警/定时抓拍 -> 云评估 -> 案件 -> 报告。

## 19. 明确不做的事

第一版先不做以下内容：

- 重写 CosmoEdge 底层 Flow Runtime。
- 用 YAML/DSL 替换现有运行时编排格式。
- 引入持续视频上云分析。
- 让 LLM 直接生成并执行底层变更命令。
- 自动触发物理处置、工单或联动控制。
- 多设备集中编排平台。

这些都可能以后做，但不应该阻塞第一版。

## 20. 结论

最稳妥的路径不是“把 CosmoEdge 改造成 Agent 框架”，而是“把 CosmoEdge 作为可靠执行核，外接一个 agentic 控制面”。

第一版成功标准不是界面多炫，也不是模型多聪明，而是下面四件事同时成立：

1. 能把自然语言需求稳定收敛成结构化 Spec。
2. 能把 Spec 稳定编译成现有引擎可执行配置。
3. 能让所有高风险变更都经过审批和审计。
4. 能把告警和定时巡检闭环成结构化风险案件与报告。

只要这四件事先跑通，后续无论要不要升级 UI、扩展云协同、增加多设备治理，都会容易很多。
