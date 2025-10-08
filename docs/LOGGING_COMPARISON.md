# Logging Comparison: Before vs After

This document shows the difference between standard logging and Rich logging in the ReasoningAgent framework.

## Before: Standard Loguru Logging

### Console Output (Plain Text)
```
2025-10-07 14:32:15 | INFO     | agent:run - [RUN] Starting task: What is 42 * 17?
2025-10-07 14:32:15 | DEBUG    | agent:_reasoning_call - [REASONING] Input text: What is 42 * 17?
2025-10-07 14:32:16 | DEBUG    | agent:_execute_task_with_tool - [EXECUTION] Task: Calculate 42 * 17
2025-10-07 14:32:16 | DEBUG    | agent:_execute_task_with_tool - [EXECUTION] Tool: calculator
2025-10-07 14:32:16 | INFO     | agent:run - [RUN] Task completed successfully in 1 iteration(s)
2025-10-07 14:32:16 | INFO     | agent:run - [METRICS] {'total_iterations': 1, 'total_tokens': 1245, 'execution_time_seconds': 2.34, 'llm_calls': 3, 'tool_executions': 1, 'successful_tool_calls': 1, 'failed_tool_calls': 0, 'task_completed': True, 'avg_tokens_per_llm_call': 415.0, 'success_rate': 1.0}
```

### Error Output (Plain Text)
```
2025-10-07 14:35:22 | ERROR    | agent:_execute_task_with_tool - [EXECUTION] Error executing tool calculator: division by zero
Traceback (most recent call last):
  File "/path/to/agent.py", line 642, in _execute_task_with_tool
    result = tool.run(tool_args)
  File "/path/to/tools.py", line 89, in run
    return eval(expression)
ZeroDivisionError: division by zero
```

---

## After: Rich Enhanced Logging

### Console Output (Colored & Formatted)

#### Startup Message
```
╭────────────────────────────────── 🚀 Welcome ───────────────────────────────────╮
│                         ReasoningAgent Framework                               │
│                  Demonstration of Rich Logging Features                        │
╰────────────────────────────────────────────────────────────────────────────────╯
```

#### Configuration Display
```
⚙️  Agent Configuration
└── LLM
    ├── api_base: http://localhost:11434/v1
    ├── model: gemma3:27b
    └── temperature: 0.7
└── Agent
    ├── verbose: true
    └── max_iterations: 10
```

#### Available Tools
```
                              🛠️  Available Tools
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tool           ┃ Description                                                 ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ search         │ Search for information on the web                           │
│ calculator     │ Perform mathematical calculations                           │
│ file_reader    │ Read contents of a file                                     │
│ shell_command  │ Execute shell commands                                      │
│ api_request    │ Make HTTP API requests                                      │
└────────────────┴─────────────────────────────────────────────────────────────┘
```

#### Execution Logs (Colored)
```
[10/07/25 14:32:15] INFO     [bold blue]Starting agent initialization...[/bold blue]
                    INFO     [green]✓[/green] Agent initialized successfully
```

#### Test Query Panel
```
╭─────────────────────────────────── 📝 Test Query ────────────────────────────────╮
│ What is 42 * 17?                                                                │
╰─────────────────────────────────────────────────────────────────────────────────╯
```

#### Reasoning Phase Panel
```
╭────────────────────────────── 🧠 Reasoning Phase ────────────────────────────────╮
│ The user wants to calculate the product of 42 and 17. I will use the            │
│ calculator tool to compute this.                                                 │
│                                                                                  │
│ Planned Tasks:                                                                   │
│   1. Calculate 42 multiplied by 17 (Tool: calculator)                           │
╰─────────────────────────────────────────────────────────────────────────────────╯
```

#### Metrics Table
```
                          🤖 Agent Execution Metrics
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric             ┃ Value     ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Total Iterations   │ 1         │
│ Total Tokens       │ 1,245     │
│ Execution Time     │ 2.34s     │
│ LLM Calls          │ 3         │
│ Tool Executions    │ 1         │
│ Successful Tools   │ 1         │
│ Failed Tools       │ 0         │
│ Task Completed     │ ✅ Yes     │
│ Avg Tokens/Call    │ 415.0     │
│ Success Rate       │ 100.0%    │
└────────────────────┴───────────┘
```

#### Execution History Table
```
                              📊 Execution History
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Iteration ┃ Task                            ┃ Tool       ┃ Status    ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ 1         │ Calculate 42 multiplied by 17   │ calculator │ completed │
└───────────┴─────────────────────────────────┴────────────┴───────────┘
```

#### Result Panel
```
╭──────────────────────────────────── ✅ Result ───────────────────────────────────╮
│ 42 multiplied by 17 equals 714                                                  │
╰─────────────────────────────────────────────────────────────────────────────────╯
```

#### Completion Status Tree
```
🎯 Completion Status
└── Complete: ✅ Yes
    ├── Reasoning: The calculation has been completed successfully
    └── Next Action: N/A
```

### Error Output (Rich Tracebacks)

#### Enhanced Traceback with Syntax Highlighting
```
╭─────────────────────────────── Traceback (most recent call last) ───────────────────────────────╮
│ /path/to/agent.py:642 in _execute_task_with_tool                                                │
│                                                                                                   │
│   639 │   try:                                                                                   │
│   640 │   │   logger.debug(f"[EXECUTION] Executing tool {task.tool_name} with args: {tool_args}")│
│   641 │   │                                                                                       │
│ ❱ 642 │   │   result = tool.run(tool_args)                                                       │
│   643 │   │   logger.debug(f"[EXECUTION] Tool result: {result}")                                 │
│   644 │   │                                                                                       │
│   645 │   │   self.tracer.set_attribute("tool.result", str(result)[:500])                        │
│                                                                                                   │
│ ╭──────────────────────────────── locals ────────────────────────────────╮                      │
│ │   self = <ReasoningAgent object at 0x7f8b1c2d4e50>                     │                      │
│ │   task = TaskExecution(description='Calculate 100/0', ...)             │                      │
│ │   tool = CalculatorTool(name='calculator', ...)                        │                      │
│ │   tool_args = {'expression': '100/0'}                                  │                      │
│ ╰─────────────────────────────────────────────────────────────────────────╯                     │
│                                                                                                   │
│ /path/to/tools.py:89 in run                                                                      │
│                                                                                                   │
│   86 │   def run(self, args: CalculatorInput) -> str:                                            │
│   87 │   │   """Execute calculation."""                                                          │
│   88 │   │   expression = args.expression                                                        │
│ ❱ 89 │   │   return eval(expression)                                                             │
│   90 │                                                                                            │
│                                                                                                   │
│ ╭──────────────────────────────── locals ────────────────────────────────╮                      │
│ │   self = <CalculatorTool object at 0x7f8b1c3e5a90>                     │                      │
│ │   args = CalculatorInput(expression='100/0')                           │                      │
│ │   expression = '100/0'                                                 │                      │
│ ╰─────────────────────────────────────────────────────────────────────────╯                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
ZeroDivisionError: division by zero
```

---

## Key Improvements

### 1. Visual Hierarchy
- **Before**: Everything looks the same
- **After**: Clear visual separation with panels, tables, and colors

### 2. Readability
- **Before**: Dense text, hard to scan
- **After**: Structured data, easy to find information

### 3. Error Debugging
- **Before**: Basic traceback
- **After**: Syntax-highlighted code, local variables, better context

### 4. Metrics Display
- **Before**: JSON blob on one line
- **After**: Formatted table with emojis and formatted numbers

### 5. Data Presentation
- **Before**: Plain text lists
- **After**: Tables, trees, and panels for different data types

### 6. Color Coding
- **Before**: No colors
- **After**:
  - 🟢 Green for success
  - �� Blue for info
  - 🟡 Yellow for warnings
  - 🔴 Red for errors

### 7. Progress Indication
- **Before**: No visual feedback
- **After**: Checkmarks (✅/❌), emojis, and status indicators

### 8. Professional Output
- **Before**: Developer logs only
- **After**: Presentation-ready output

---

## Usage Comparison

### Before (Standard Logging)
```python
from loguru import logger

logger.info(f"Agent metrics: {metrics}")
```

**Output:**
```
2025-10-07 14:32:16 | INFO | Agent metrics: {'total_tokens': 1245, 'execution_time': 2.34, ...}
```

### After (Rich Logging)
```python
from linus.agents.logging_config import log_metrics

log_metrics(metrics, title="Agent Execution Metrics")
```

**Output:**
```
         Agent Execution Metrics
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric          ┃ Value    ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Total Tokens    │ 1,245    │
│ Execution Time  │ 2.34s    │
│ Success Rate    │ 100.0%   │
└─────────────────┴──────────┘
```

---

## Migration Guide

### Step 1: Update Imports
```python
# Old
from loguru import logger

# New (add Rich utilities)
from loguru import logger
from linus.agents.logging_config import (
    setup_rich_logging,
    log_with_panel,
    log_metrics
)
```

### Step 2: Setup Rich Logging
```python
# Initialize at startup
console = setup_rich_logging(
    level="INFO",
    log_file="logs/app.log"
)
```

### Step 3: Use Rich Features
```python
# Standard logs still work
logger.info("Processing request")

# Enhanced with markup
logger.info("[bold green]✓[/bold green] Request processed")

# Use utilities for structured data
log_metrics(metrics, title="Performance")
log_with_panel("Important message", title="Alert")
```

---

## Conclusion

Rich logging provides:
- ✅ Better readability
- ✅ Faster debugging
- ✅ Professional output
- ✅ Same API (loguru still works)
- ✅ Backward compatible
- ✅ Easy to adopt

Try it yourself: `python example_rich_logging.py`
