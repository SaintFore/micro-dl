# micro-dl 领域语言

micro-dl 是一个兼顾教学清晰度与真实训练价值的深度学习框架。本术语表固定
重构期间使用的领域语言，不承担架构规格或实现说明。

## 计算

**Tensor**:
具有身份、Storage 与 autograd 状态的多维数据对象。
_Avoid_: Array、data

**Storage**:
由 Tensor 私有拥有、且属于某个 Backend 的底层数组资源。
_Avoid_: data、底层数据

**Operator**:
定义一次 Tensor 计算及其 forward/backward 数学语义的无状态运算。
_Avoid_: 算法函数、backend operator

**GradFn**:
保存某次 Operator 调用的 parents 与反向规则的图节点行为。
_Avoid_: BackwardFn、backward function

**Computation Graph**:
由 Tensor 与 GradFn 在 forward 中动态形成的计算依赖图。
_Avoid_: Model graph、静态图

**Leaf Tensor**:
由用户或模型状态直接创建、不是某次 Operator 结果的 Tensor。
_Avoid_: Input node

## 模型状态

**Module**:
持有 Parameter、Buffer 和子 Module，并通过组合 Operators 表达模型行为的对象。
_Avoid_: Layer base、模型容器

**Parameter**:
参与优化并跨训练步骤持久存在的可训练 Tensor。
_Avoid_: Weight（泛指时）、可训练 Buffer

**Buffer**:
不参与优化、但属于 Module 持久状态的 Tensor。
_Avoid_: Constant、cache

**Cache**:
只服务一次 forward/backward 或一次内部计算、不属于持久模型状态的临时数据。
_Avoid_: Buffer、state

## 执行与训练

**Runtime**:
Backend implementation 实际调用的数组执行环境；首批 Runtime 是 NumPy CPU 与
CuPy CUDA。
_Avoid_: Backend、数组工具类

**Backend**:
Tensor 所属的、可替换 Runtime implementation 的内部能力 seam。Backend 决定
Storage 的创建、识别、转换与数组计算语义。
_Avoid_: Runtime、Device、任意 xp 模块

**Trainer**:
统一编排训练与验证生命周期的对象。
_Avoid_: Training loop、Experiment

**Experiment**:
选择数据、模型、loss、optimizer、scheduler 和训练参数的可运行装配。
_Avoid_: Trainer、Example infrastructure
