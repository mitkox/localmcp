#!/usr/bin/env python
import json
import sqlite3
import logging
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, Union, Literal
import uvicorn
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel, Field, AnyUrl, ValidationError
from llama_cpp import Llama
from dotenv import load_dotenv

# -------------------------------
# Configuration and Logging Setup
# -------------------------------

load_dotenv()  # Load environment variables from .env file

class Settings(BaseModel):
    model_path: str = Field(..., env="LLAMA_MODEL_PATH")
    db_path: str = Field("timeseries.db", env="SQLITE_DB_PATH")
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    max_tokens: int = Field(128, env="MAX_TOKENS")
    allowed_tables: set = Field({"sensor_data", "metrics", "manufacturing_data"}, env="ALLOWED_TABLES")

settings = Settings(
    model_path=os.getenv("LLAMA_MODEL_PATH", "/Users/mitko/dev/models/DeepHermes-3-Llama-3-8B-q8.gguf"),
    db_path=os.getenv("SQLITE_DB_PATH", "timeseries.db"),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MCP-Server")

# -------------------------------
# Security and Validation Helpers
# -------------------------------

class SQLInjectionError(ValueError):
    """Raised when potential SQL injection is detected"""

def validate_sql_identifier(identifier: str) -> None:
    """Validate SQL identifiers to prevent injection attacks"""
    if not identifier.isidentifier() or any(c in identifier for c in ";-'\"\\"):
        logger.error(f"Potential SQL injection detected: {identifier}")
        raise SQLInjectionError(f"Invalid identifier: {identifier}")

# -------------------------------
# Tool System Infrastructure
# -------------------------------

class ToolRegistry:
    """Registry for managing available MCP tools"""
    def __init__(self):
        self.tools: Dict[str, Dict] = {}

    def register(self, func) -> None:
        """Register a tool with metadata"""
        self.tools[func._tool_name] = {
            "function": func,
            "schema": func._args_schema.model_json_schema(),
            "description": func.__doc__
        }

    def get_tool(self, name: str) -> Optional[Dict]:
        """Retrieve tool metadata"""
        return self.tools.get(name)

tool_registry = ToolRegistry()

def tool(tool_name: str, args_schema):
    """Enhanced tool decorator with auto-registration"""
    def decorator(func):
        func._tool_name = tool_name
        func._args_schema = args_schema
        tool_registry.register(func)
        return func
    return decorator

# -------------------------------
# Time Series Tool Implementation
# -------------------------------

class SQLiteTimeSeriesInput(BaseModel):
    table: str = Field(..., description="Name of the SQLite table containing time series data")
    metric: str = Field(..., description="Column name for the numeric metric")
    start_time: datetime = Field(..., description="Start time in ISO 8601 format")
    end_time: datetime = Field(..., description="End time in ISO 8601 format")

    @classmethod
    def validate_table(cls, value):
        if value not in settings.allowed_tables:
            raise ValueError(f"Table {value} not permitted")
        return value

@tool("query_sqlite_timeseries", args_schema=SQLiteTimeSeriesInput)
def query_sqlite_timeseries(args: SQLiteTimeSeriesInput) -> Dict[str, Any]:
    """
    Advanced time series analytics with multiple metrics
    
    Returns:
        Dict containing statistics and query metadata
    """
    try:
        validate_sql_identifier(args.table)
        metric_expr = args.metric
        
        # Skip validation for CASE expressions
        if not metric_expr.upper().startswith('CASE'):
            validate_sql_identifier(metric_expr)
        
        with sqlite3.connect(settings.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = f"""
                SELECT 
                    AVG({metric_expr}) as avg_value,
                    MIN({metric_expr}) as min_value,
                    MAX({metric_expr}) as max_value,
                    SUM({metric_expr}) as sum_value,
                    COUNT(*) as sample_count
                FROM {args.table}
                WHERE timestamp BETWEEN ? AND ?
            """
            
            cursor.execute(query, (
                args.start_time.isoformat(),
                args.end_time.isoformat()
            ))
            
            result = dict(cursor.fetchone())
            return {
                "metadata": {
                    "query": query,
                    "params": [args.start_time.isoformat(), args.end_time.isoformat()],
                    "execution_time": datetime.utcnow().isoformat()
                },
                "statistics": result
            }
            
    except (sqlite3.Error, SQLInjectionError) as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Response Models
# -------------------------------

class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None

class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: str
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None
    timestamp: datetime

# -------------------------------
# Enhanced JSON-RPC System
# -------------------------------

class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"]
    method: str
    params: Dict[str, Any]
    id: str

class ToolDiscoveryResponse(BaseModel):
    available_tools: Dict[str, Dict]
    api_version: str
    model_info: Dict

# -------------------------------
# FastAPI Application Setup
# -------------------------------

app = FastAPI(
    title="Enterprise MCP Analytics Gateway",
    description="Secure integration platform combining LLM capabilities with time series analytics",
    version="2.0.0",
    openapi_tags=[{
        "name": "MCP",
        "description": "Core Model Context Protocol endpoints"
    }]
)

@app.on_event("startup")
async def initialize_model():
    """Initialize the LLM model with enhanced configuration"""
    global llm
    try:
        llm = Llama(
            model_path=settings.model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            temperature=0.7,
            max_tokens=4096,  # Increased for longer responses
            logits_all=True,
            verbose=True,
            n_threads=8  # Optimize performance
        )
        logger.info(f"Model {settings.model_path} loaded successfully")
    except Exception as e:
        logger.critical(f"Model initialization failed: {str(e)}")
        raise RuntimeError("Model loading failed")

# -------------------------------
# Enhanced API Endpoints
# -------------------------------

@app.post("/api/v1/mcp", tags=["MCP"], response_model=JSONRPCResponse)
async def handle_mcp_request(request: JSONRPCRequest) -> JSONRPCResponse:
    """Enterprise-grade MCP request handler with enhanced security"""
    try:
        if request.method == "mcp.get_information":
            if "prompt" not in request.params:
                return error_response(request.id, -32602, "Missing required parameter: prompt")
            try:
                result = await run_async_inference(request.params["prompt"])
                if not result:
                    return error_response(request.id, -32603, "Model returned empty response")
                return create_response(request.id, result)
            except HTTPException as e:
                error_msg = str(e.detail) if isinstance(e.detail, (str, dict)) else str(e)
                return error_response(
                    request.id,
                    -32603,
                    error_msg
                )
            except Exception as e:
                logger.error(f"Inference failed: {str(e)}", exc_info=True)
                return error_response(
                    request.id,
                    -32603,
                    f"Model inference failed: {str(e)}"
                )
                
        elif request.method == "mcp.tool_call":
            if "tool" not in request.params:
                return error_response(request.id, -32602, "Missing required parameter: tool")
            tool_result = execute_tool_safely(
                request.params["tool"],
                request.params.get("args", {})
            )
            return create_response(request.id, tool_result)
        
        elif request.method == "mcp.discover":
            tools_info = {
                name: {
                    "description": tool["description"],
                    "schema": tool["schema"]
                }
                for name, tool in tool_registry.tools.items()
            }
            return create_response(request.id, {
                "tools": tools_info,
                "capabilities": ["time_series", "llm_inference"],
                "version": "2.0.0"
            })
        else:
            return error_response(request.id, -32601, f"Method {request.method} not found")
            
    except ValidationError as e:
        return error_response(request.id, -32602, f"Invalid params: {str(e)}")
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}", exc_info=True)
        return error_response(request.id, -32603, f"Internal error: {str(e)}")

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Advanced system health monitoring endpoint"""
    checks = {
        "database_connected": False,
        "model_loaded": False,
        "active_workers": 0
    }
    
    try:
        with sqlite3.connect(settings.db_path) as conn:
            checks["database_connected"] = True
    except Exception as e:
        logger.warning(f"Database health check failed: {str(e)}")
    
    checks["model_loaded"] = hasattr(llm, "tokenize")
    
    return {
        "status": "OK" if all(checks.values()) else "DEGRADED",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }

# -------------------------------
# Support Functions
# -------------------------------

async def run_async_inference(prompt: str) -> str:
    """Execute model inference with async wrapper"""
    try:
        logger.info(f"Running inference with prompt: {prompt[:100]}...")
        
        if not hasattr(llm, "tokenize"):
            logger.error("Model not properly initialized")
            raise ValueError("LLM model not initialized")
        
        # Add system prompt to ensure complete responses
        system_prompt = "You are a manufacturing analytics expert. Provide complete, thorough responses. Always end your analysis with a clear conclusion."
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        result = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: llm(
                full_prompt,
                max_tokens=4096,  # Match model config
                stop=["</s>", "\n\n\n"],  # Adjusted stop tokens
                echo=False,
                temperature=0.7,
                top_p=0.9,  # Add top_p sampling
                repeat_penalty=1.1  # Prevent repetitive outputs
            )
        )
        
        # Validate response structure
        if not result:
            raise ValueError("Empty response from LLM")
            
        if not isinstance(result, dict):
            raise ValueError(f"Invalid response type: {type(result)}")
            
        if "choices" not in result or not result["choices"]:
            raise ValueError("No choices in LLM response")
            
        response_text = result["choices"][0].get("text", "").strip()
        if not response_text:
            raise ValueError("Empty text in response")
        
        # Ensure response completeness
        if not any(response_text.rstrip().endswith(char) for char in ".!?"):
            response_text += "."
            
        logger.info(f"Successfully generated response of length {len(response_text)}")
        return response_text

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        error_detail = {
            "message": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )

def execute_tool_safely(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Safe tool execution with validation"""
    if not (tool := tool_registry.get_tool(tool_name)):
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    
    try:
        # Convert datetime strings to datetime objects if needed
        if 'start_time' in args:
            args['start_time'] = datetime.fromisoformat(args['start_time'])
        if 'end_time' in args:
            args['end_time'] = datetime.fromisoformat(args['end_time'])
            
        # Create validated input object and execute tool
        validated_args = SQLiteTimeSeriesInput(**args)
        return tool["function"](validated_args)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e.errors()))
    except Exception as e:
        logger.error(f"Tool execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

def create_response(request_id: str, result: Any) -> JSONRPCResponse:
    """Standardized response formatting"""
    return JSONRPCResponse(
        id=request_id,
        result=result,
        timestamp=datetime.utcnow()
    )

def error_response(request_id: str, code: int, message: str) -> JSONRPCResponse:
    """Enhanced error reporting"""
    return JSONRPCResponse(
        id=request_id,
        error=JSONRPCError(
            code=code,
            message=message,
            data={"timestamp": datetime.utcnow()}
        ),
        timestamp=datetime.utcnow()
    )

# -------------------------------
# Enterprise Execution Setup
# -------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        timeout_keep_alive=120,
        log_config=None, # Use default logging configuration
    )