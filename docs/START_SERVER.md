# Starting the FastAPI Server

## Important: Activate Conda Environment First!

Before running any Python commands or starting the server, always activate the `agents` conda environment:

```bash
conda activate agents
```

## Starting the Server

From the project root directory:

```bash
# Activate environment
conda activate agents

# Start the server
cd src
python app.py
```

Or using uvicorn directly:

```bash
conda activate agents
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

## Verifying Environment

Check that you're in the correct environment:

```bash
echo $CONDA_DEFAULT_ENV
# Should show: agents

# Verify packages are installed
pip list | grep -E "(langfuse|openai|python-dotenv)"
```

## Environment Variables

The server loads environment variables from `src/.env`. Key variables for tracing:

```
TELEMETRY_ENABLED=true
TELEMETRY_EXPORTER=langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

## Checking Traces

After making requests, traces should appear in Langfuse at:
- **Local**: http://localhost:3000
- Look for traces named `agent_run`

## Troubleshooting

If traces don't appear:
1. Verify conda environment is active: `echo $CONDA_DEFAULT_ENV`
2. Check packages installed: `pip list | grep langfuse`
3. Check logs for `[LANGFUSE]` messages
4. Verify Langfuse is running: `curl http://localhost:3000/api/public/health`
5. Check environment variables are loaded correctly
