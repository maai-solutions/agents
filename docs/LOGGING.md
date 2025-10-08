# Logging Configuration

## Overview

The agent system uses [Loguru](https://github.com/Delgan/loguru) for comprehensive logging with automatic rotation, retention, and multiple output formats.

## Configuration

### Location

Logging is configured in `src/app.py` (lines 20-37):

```python
# Configure logging early
os.makedirs("logs", exist_ok=True)

# Remove default handler and add custom ones
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/agent_api.log",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
    level="DEBUG"
)
```

### Outputs

The system logs to **two destinations**:

#### 1. Console (stderr)
- **Level**: INFO and above
- **Format**: Colored output with timestamps
- **Purpose**: Real-time monitoring during development

#### 2. File (logs/agent_api.log)
- **Level**: DEBUG and above (captures everything)
- **Format**: Plain text with timestamps
- **Rotation**: Automatically rotates when file reaches 10 MB
- **Retention**: Keeps logs for 7 days
- **Compression**: Old logs are compressed as .zip files

## Log Levels

From most to least verbose:

1. **DEBUG** - Detailed diagnostic information (file only)
2. **INFO** - General informational messages (console + file)
3. **WARNING** - Warning messages (console + file)
4. **ERROR** - Error messages (console + file)
5. **CRITICAL** - Critical errors (console + file)

## Log File Structure

```
logs/
├── agent_api.log              # Current log file
├── agent_api.2024-10-07.log.zip  # Rotated and compressed
└── agent_api.2024-10-06.log.zip  # Older rotated logs
```

## Log Format

### Console Format
```
2024-10-07 17:52:57 | INFO     | app:lifespan - Starting ReasoningAgent API...
```

### File Format (same structure, no colors)
```
2024-10-07 17:52:57 | INFO     | app:lifespan - Starting ReasoningAgent API...
2024-10-07 17:52:57 | DEBUG    | agent:run - Processing query: Calculate 42 * 17
```

**Format Fields**:
- **Timestamp**: `YYYY-MM-DD HH:mm:ss`
- **Level**: Log level (INFO, DEBUG, etc.)
- **Location**: `module:function`
- **Message**: The actual log message

## Usage in Code

### Basic Logging

```python
from loguru import logger

logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical failure")
```

### With Variables

```python
logger.info(f"Processing query: {query}")
logger.debug(f"Agent state: {agent.state}")
logger.error(f"Failed to execute tool {tool_name}: {error}")
```

### With Exception Tracing

```python
try:
    result = agent.run(query)
except Exception as e:
    logger.error(f"Error processing query: {e}", exc_info=True)
```

The `exc_info=True` parameter includes the full stack trace in the logs.

## Viewing Logs

### Real-time (Console)

When running the app, INFO and above messages are displayed in the console with colors:

```bash
python src/app.py
```

### View Log File

```bash
# View entire log
cat logs/agent_api.log

# View last 50 lines
tail -50 logs/agent_api.log

# Follow in real-time
tail -f logs/agent_api.log

# Search for errors
grep ERROR logs/agent_api.log

# Search for specific query
grep "Processing query" logs/agent_api.log
```

## Log Rotation

Logs automatically rotate when:
- **File size** exceeds 10 MB
- **Retention period** is 7 days (older logs are deleted)

Example:
```
logs/agent_api.log             # 9.8 MB (current)
logs/agent_api.2024-10-07.log.zip  # 3.2 MB (1 day old)
logs/agent_api.2024-10-06.log.zip  # 4.1 MB (2 days old)
# Logs older than 7 days are automatically deleted
```

## Configuration Options

You can modify the logging configuration in `src/app.py`:

### Change Log Level

```python
# More verbose (show DEBUG in console too)
logger.add(sys.stderr, level="DEBUG")

# Less verbose (only show warnings and errors)
logger.add(sys.stderr, level="WARNING")
```

### Change Rotation

```python
# Rotate every day at midnight
logger.add("logs/agent_api.log", rotation="00:00")

# Rotate every 50 MB
logger.add("logs/agent_api.log", rotation="50 MB")

# Rotate every week
logger.add("logs/agent_api.log", rotation="1 week")
```

### Change Retention

```python
# Keep logs for 30 days
logger.add("logs/agent_api.log", retention="30 days")

# Keep last 10 files
logger.add("logs/agent_api.log", retention=10)
```

### Disable Compression

```python
# Don't compress rotated logs
logger.add("logs/agent_api.log", compression=None)
```

## Example Log Output

```
2024-10-07 17:52:57 | INFO     | app:lifespan - Starting ReasoningAgent API...
2024-10-07 17:52:57 | INFO     | app:lifespan - Agent initialized with 6 tools
2024-10-07 17:52:57 | INFO     | app:lifespan - Using model: gemma3:27b at http://localhost:11434/v1
2024-10-07 17:53:05 | DEBUG    | agent:_reasoning_call - Starting reasoning phase
2024-10-07 17:53:06 | DEBUG    | agent:_execute_task_with_tool - Executing tool: calculator
2024-10-07 17:53:06 | INFO     | agent:run - Task completed in 1.2s after 3 iterations
2024-10-07 17:53:11 | WARNING  | agent:_check_completion - Max iterations reached
2024-10-07 17:53:15 | ERROR    | app:query_agent - Error processing query: Connection timeout
```

## Best Practices

### 1. Use Appropriate Log Levels

```python
# ✅ Good
logger.debug(f"Intermediate calculation: {value}")
logger.info(f"Task completed successfully")
logger.warning(f"Unusual condition detected: {condition}")
logger.error(f"Failed to connect to LLM: {error}")

# ❌ Bad
logger.info(f"Variable x = {x}")  # Too verbose, use DEBUG
logger.error(f"Task completed")    # Not an error, use INFO
```

### 2. Include Context

```python
# ✅ Good
logger.error(f"Tool '{tool_name}' failed with args {args}: {error}")

# ❌ Bad
logger.error("Tool failed")
```

### 3. Use Exception Info for Errors

```python
# ✅ Good
try:
    result = process()
except Exception as e:
    logger.error(f"Processing failed: {e}", exc_info=True)

# ❌ Bad
except Exception as e:
    logger.error(f"Error: {e}")  # Missing stack trace
```

### 4. Don't Log Sensitive Data

```python
# ✅ Good
logger.info(f"User authenticated: user_id={user_id}")

# ❌ Bad
logger.info(f"Login: username={username}, password={password}")
```

## Troubleshooting

### Log File is Empty

If `logs/agent_api.log` exists but is empty:

1. **Check if app has run**: Logs are only written when the app runs
2. **Check log level**: Ensure messages meet the DEBUG threshold
3. **Check permissions**: Ensure write access to `logs/` directory

```bash
# Test logging manually
python -c "
from loguru import logger
logger.add('logs/test.log')
logger.info('Test message')
"
cat logs/test.log
```

### Logs Not Appearing in Console

If logs don't show in console but appear in file:

1. **Check console level**: Default is INFO, won't show DEBUG
2. **Check stderr redirection**: Ensure stderr is not redirected

### Too Much Log Output

If logs are too verbose:

```python
# Reduce console verbosity
logger.remove()
logger.add(sys.stderr, level="WARNING")  # Only warnings and errors
logger.add("logs/agent_api.log", level="DEBUG")  # Keep detailed file logs
```

## Integration with FastAPI

The logging is automatically available throughout the FastAPI app:

```python
from loguru import logger

@app.get("/endpoint")
async def endpoint():
    logger.info("Endpoint called")
    try:
        result = process()
        logger.debug(f"Result: {result}")
        return result
    except Exception as e:
        logger.error(f"Endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

FastAPI's own logs (HTTP requests, etc.) are handled separately by uvicorn.

## Summary

- ✅ **Dual output**: Console (INFO+) and file (DEBUG+)
- ✅ **Automatic rotation**: At 10 MB
- ✅ **Automatic cleanup**: 7-day retention
- ✅ **Compression**: Old logs compressed as .zip
- ✅ **Structured format**: Timestamp, level, location, message
- ✅ **Easy to use**: `from loguru import logger`
- ✅ **Production-ready**: Configured for long-running services

For more information, see the [Loguru documentation](https://loguru.readthedocs.io/).
