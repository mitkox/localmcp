# LocalMCP

A local Model Context Protocol (MCP) server implementation with time series analytics capabilities and LLM integration.

## Features

- JSON-RPC API for standardized communication
- Time series data analysis with SQLite backend
- LLM integration using llama.cpp
- Manufacturing metrics analysis and monitoring
- Enterprise-grade error handling and logging

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
LLAMA_MODEL_PATH=/path/to/your/model.gguf
SQLITE_DB_PATH=timeseries.db
```

3. Run the server:
```bash
python mcp_server.py
```

4. Run the test script:
```bash
python test_mcp.py
```

## API Endpoints

- `/api/v1/mcp` - Main JSON-RPC endpoint
- `/health` - Health check endpoint

## Available Methods

- `mcp.get_information` - Get LLM analysis
- `mcp.tool_call` - Execute analytics tools
- `mcp.discover` - List available tools and capabilities