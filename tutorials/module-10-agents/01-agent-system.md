---
layout: page
title: "Module 10: Agent系统 — 智能体"
---
# Module 10: Agent系统 — 智能体

## 理论部分

### 10.1 什么是Agent？

Agent = LLM + 工具 + 规划 + 记忆

```
┌─────────────────────────────────────────┐
│              Agent 架构                   │
│                                          │
│  ┌──────┐    ┌──────────┐    ┌───────┐  │
│  │ 规划 │───→│ LLM核心  │───→│ 执行  │  │
│  │(Plan)│    │(Reasoning)│    │(Action)│ │
│  └──────┘    └──────────┘    └───────┘  │
│      ↑            │             │        │
│      │            ↓             ↓        │
│  ┌──────┐    ┌──────────┐    ┌───────┐  │
│  │ 记忆 │    │   工具   │    │ 环境  │  │
│  │(Memory)│   │ (Tools)  │    │(Env)  │  │
│  └──────┘    └──────────┘    └───────┘  │
└─────────────────────────────────────────┘
```

### 10.2 ReAct模式 (Reasoning + Acting)

ReAct让模型交替进行推理和行动：

```
思考(Thought): 我需要查一下当前天气
行动(Action): search("北京天气")
观察(Observation): 北京今天晴，25°C
思考(Thought): 已有天气信息，可以回答
行动(Action): finish("北京今天晴，25°C")
```

### 10.3 工具调用 (Function Calling)

现代LLM支持结构化工具调用：

```json
{
  "tool": "search",
  "parameters": {
    "query": "Python教程"
  }
}
```

常见工具：
- 搜索 (Google/Bing)
- 代码执行 (Python REPL)
- 数据库查询
- API调用
- 文件读写

### 10.4 规划策略

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| 单步 | 直接执行 | 简单任务 |
| 多步链式 | 逐步推理 | 中等复杂度 |
| 任务分解 | 分解为子任务 | 复杂任务 |
| 反思 | 执行后检查修正 | 需要准确性 |

### 10.5 记忆系统

```
短期记忆: 当前对话上下文
长期记忆: 向量数据库存储的历史信息
工作记忆: 当前任务的中间结果
```

## 实践部分

### 实践1：ReAct Agent实现

```python
import json
from typing import Dict, List, Any, Callable

print("=" * 60)
print("实践1: ReAct Agent实现")
print("=" * 60)

class Tool:
    """工具基类"""
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def __call__(self, **kwargs):
        return self.func(**kwargs)

class Agent:
    """ReAct Agent"""
    def __init__(self, tools: List[Tool], max_steps=10):
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.history = []

    def get_tool_descriptions(self):
        """获取工具描述"""
        descs = []
        for name, tool in self.tools.items():
            descs.append(f"- {name}: {tool.description}")
        return "\n".join(descs)

    def think(self, query: str) -> str:
        """推理步骤 (模拟LLM)"""
        # 实际中这里调用LLM
        prompt = f"""你是一个AI助手，可以使用以下工具:
{self.get_tool_descriptions()}

用户问题: {query}

请按以下格式回复:
思考: [你的推理过程]
行动: [工具名](参数)
或
思考: [你的推理过程]
回答: [最终回答]"""
        return prompt

    def act(self, action_str: str) -> str:
        """执行工具"""
        # 解析行动字符串
        try:
            # 简单解析: tool_name(param1=value1, param2=value2)
            if '(' in action_str:
                tool_name = action_str.split('(')[0].strip()
                params_str = action_str.split('(')[1].rstrip(')')
                params = {}
                for pair in params_str.split(','):
                    if '=' in pair:
                        k, v = pair.split('=', 1)
                        params[k.strip()] = v.strip().strip('"').strip("'")
            else:
                tool_name = action_str.strip()
                params = {}

            if tool_name in self.tools:
                result = self.tools[tool_name](**params)
                return f"观察: {result}"
            else:
                return f"观察: 工具 '{tool_name}' 不存在"
        except Exception as e:
            return f"观察: 执行错误 - {str(e)}"

    def run(self, query: str) -> str:
        """运行Agent"""
        print(f"\n用户: {query}")
        print("-" * 40)

        self.history = [{"role": "user", "content": query}]

        for step in range(self.max_steps):
            # 思考
            thought_prompt = self.think(query)
            # 模拟LLM输出 (实际中调用LLM API)
            thought, action = self._simulate_llm_response(query, step)

            print(f"\n步骤 {step + 1}:")
            print(f"  思考: {thought}")
            print(f"  行动: {action}")

            if action.startswith("finish") or action.startswith("回答"):
                answer = action.split(":", 1)[1].strip() if ":" in action else action
                print(f"\n最终回答: {answer}")
                return answer

            observation = self.act(action)
            print(f"  {observation}")

            self.history.append({"role": "assistant", "content": f"{thought}\n{action}\n{observation}"})

        return "达到最大步数限制"

    def _simulate_llm_response(self, query: str, step: int):
        """模拟LLM响应 (实际中替换为真实LLM调用)"""
        # 简单的规则匹配模拟
        if "天气" in query and step == 0:
            return ("我需要查询天气信息", "get_weather(city=北京)")
        elif "天气" in query and step == 1:
            return ("已获得天气信息", "回答: 北京今天晴，温度25°C，适合出行。")
        elif "计算" in query or "数学" in query:
            return ("需要执行计算", "calculate(expression=2+3*4)")
        elif step == 0:
            return ("我需要搜索相关信息", "search(query=Python教程)")
        else:
            return ("已获取足够信息", "回答: 根据搜索结果，Python是一种广泛使用的高级编程语言...")

# 定义工具
def get_weather(city="北京"):
    """获取天气"""
    weather_data = {
        "北京": "晴，25°C",
        "上海": "多云，28°C",
        "深圳": "雷阵雨，30°C",
    }
    return weather_data.get(city, "未知城市")

def search(query=""):
    """搜索信息"""
    return f"找到关于'{query}'的10条结果"

def calculate(expression=""):
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except:
        return "计算错误"

# 创建Agent
tools = [
    Tool("get_weather", "获取指定城市的天气信息", get_weather),
    Tool("search", "搜索互联网信息", search),
    Tool("calculate", "计算数学表达式", calculate),
]

agent = Agent(tools)

# 测试
agent.run("北京今天天气怎么样？")
agent.run("帮我计算2+3*4")
```

### 实践2：工具调用格式

```python
print("\n" + "=" * 60)
print("实践2: Function Calling格式")
print("=" * 60)

# OpenAI Function Calling格式
tools_schema = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "search",
        "description": "搜索信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                }
            },
            "required": ["query"]
        }
    }
]

# 模拟LLM的工具调用响应
llm_response = {
    "content": None,
    "function_call": {
        "name": "get_weather",
        "arguments": '{"city": "北京"}'
    }
}

print("LLM工具调用响应:")
print(json.dumps(llm_response, indent=2, ensure_ascii=False))

# 解析并执行
func_name = llm_response["function_call"]["name"]
func_args = json.loads(llm_response["function_call"]["arguments"])

print(f"\n执行工具: {func_name}({func_args})")
if func_name == "get_weather":
    result = get_weather(**func_args)
    print(f"结果: {result}")
```

### 实践3：多Agent协作

```python
print("\n" + "=" * 60)
print("实践3: 多Agent协作")
print("=" * 60)

class MultiAgentSystem:
    """多Agent协作系统"""
    def __init__(self):
        self.agents = {}

    def register(self, name, agent, role):
        self.agents[name] = {"agent": agent, "role": role}

    def coordinate(self, task):
        """协调多个Agent完成任务"""
        print(f"\n任务: {task}")
        print("=" * 40)

        # 1. 规划Agent分解任务
        print("\n[规划Agent] 分解任务...")
        subtasks = [
            "搜索最新AI论文",
            "总结关键发现",
            "生成报告"
        ]
        for i, st in enumerate(subtasks):
            print(f"  子任务{i+1}: {st}")

        # 2. 执行Agent执行子任务
        print("\n[执行Agent] 执行子任务...")
        results = []
        for st in subtasks:
            result = f"完成: {st}"
            results.append(result)
            print(f"  ✓ {result}")

        # 3. 审查Agent检查质量
        print("\n[审查Agent] 检查质量...")
        print("  ✓ 内容完整性: 通过")
        print("  ✓ 格式规范: 通过")
        print("  ✓ 准确性: 通过")

        return results

# 使用
system = MultiAgentSystem()
system.register("planner", None, "任务规划")
system.register("executor", None, "任务执行")
system.register("reviewer", None, "质量审查")

system.coordinate("研究生成式AI的最新进展")
```

## 总结

| 组件 | 实现方式 | 推荐 |
|------|---------|------|
| 推理 | ReAct / Plan-and-Execute | ReAct |
| 工具 | Function Calling | OpenAI格式 |
| 记忆 | 向量数据库 | Chroma/FAISS |
| 协调 | 多Agent / 单Agent+多工具 | 单Agent+多工具 |
| 规划 | Chain-of-Thought / ToT | CoT |

**下一步**: [Module 11: 模型评估](./../module-11-evaluation/)
