#!/usr/bin/env python
import sqlite3
import requests
import json
from datetime import datetime, timedelta
import random

def setup_manufacturing_db():
    with sqlite3.connect("timeseries.db") as conn:
        cursor = conn.cursor()
        
        # Create manufacturing metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS manufacturing_data (
                timestamp TEXT,
                production_rate REAL,
                quality_score REAL,
                equipment_temperature REAL,
                vibration_level REAL,
                power_consumption REAL,
                status TEXT
            )
        """)
        
        # Generate realistic manufacturing data for last 24 hours
        base_time = datetime.now() - timedelta(days=1)
        test_data = []
        
        # Simulate normal operating conditions with occasional anomalies
        for i in range(24*60):  # One reading per minute
            timestamp = base_time + timedelta(minutes=i)
            
            # Simulate production rate (units per minute, normally 8-12)
            production_rate = 10 + random.normalvariate(0, 1)
            
            # Quality score (0-100%)
            quality_score = 95 + random.normalvariate(0, 2)
            if quality_score > 100: quality_score = 100
            
            # Equipment temperature (normal range 60-80°C)
            equipment_temp = 70 + random.normalvariate(0, 5)
            
            # Vibration level (0-5 mm/s)
            vibration = 2 + random.normalvariate(0, 0.5)
            
            # Power consumption (kW)
            power = 75 + random.normalvariate(0, 5)
            
            # Equipment status
            if equipment_temp > 85:
                status = "WARNING_HIGH_TEMP"
            elif vibration > 4:
                status = "WARNING_HIGH_VIBRATION"
            elif quality_score < 90:
                status = "WARNING_LOW_QUALITY"
            else:
                status = "NORMAL"
                
            test_data.append((
                timestamp.isoformat(),
                production_rate,
                quality_score,
                equipment_temp,
                vibration,
                power,
                status
            ))
        
        cursor.executemany(
            """INSERT INTO manufacturing_data 
               (timestamp, production_rate, quality_score, equipment_temperature,
                vibration_level, power_consumption, status)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            test_data
        )
        conn.commit()

def test_manufacturing_analysis():
    # Get recent metrics
    start_time = (datetime.now() - timedelta(hours=12)).isoformat()
    end_time = datetime.now().isoformat()
    
    try:
        # Query time series data
        payload_metrics = {
            "jsonrpc": "2.0",
            "method": "mcp.tool_call",
            "params": {
                "tool": "query_sqlite_timeseries",
                "args": {
                    "table": "manufacturing_data",
                    "metric": "quality_score",
                    "start_time": start_time,
                    "end_time": end_time
                }
            },
            "id": "metrics-1"
        }
        
        # Get metrics first
        response = requests.post(
            "http://localhost:8000/api/v1/mcp",
            json=payload_metrics,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        response_data = response.json()
        
        if 'error' in response_data and response_data['error']:
            raise Exception(f"API Error: {response_data['error']}")
            
        metrics_data = response_data["result"]["statistics"]
        
        # Count warnings
        warning_query = {
            "jsonrpc": "2.0",
            "method": "mcp.tool_call",
            "params": {
                "tool": "query_sqlite_timeseries",
                "args": {
                    "table": "manufacturing_data",
                    "metric": "CASE WHEN status != 'NORMAL' THEN 1 ELSE 0 END",
                    "start_time": start_time,
                    "end_time": end_time
                }
            },
            "id": "warnings-1"
        }
        
        warnings_response = requests.post(
            "http://localhost:8000/api/v1/mcp",
            json=warning_query,
            headers={"Content-Type": "application/json"}
        )
        warnings_response.raise_for_status()
        warnings_data = warnings_response.json()
        
        if 'error' in warnings_data and warnings_data['error']:
            raise Exception(f"API Error: {warnings_data['error']}")
            
        warning_count = warnings_data["result"]["statistics"]["sum_value"]
        
        # Get additional metrics for temperature and vibration
        temp_query = {
            "jsonrpc": "2.0",
            "method": "mcp.tool_call",
            "params": {
                "tool": "query_sqlite_timeseries",
                "args": {
                    "table": "manufacturing_data",
                    "metric": "equipment_temperature",
                    "start_time": start_time,
                    "end_time": end_time
                }
            },
            "id": "temp-1"
        }
        
        vibration_query = {
            "jsonrpc": "2.0",
            "method": "mcp.tool_call",
            "params": {
                "tool": "query_sqlite_timeseries",
                "args": {
                    "table": "manufacturing_data",
                    "metric": "vibration_level",
                    "start_time": start_time,
                    "end_time": end_time
                }
            },
            "id": "vibration-1"
        }
        
        temp_response = requests.post(
            "http://localhost:8000/api/v1/mcp",
            json=temp_query,
            headers={"Content-Type": "application/json"}
        ).json()
        vibration_response = requests.post(
            "http://localhost:8000/api/v1/mcp",
            json=vibration_query,
            headers={"Content-Type": "application/json"}
        ).json()
        
        # Format prompt with actual values
        analysis_prompt = """Analyze the following manufacturing line metrics and provide a complete assessment. Structure your response with clear sections.

Current Manufacturing Metrics:
- Quality Score: {quality_score:.2f}% average (target: >95%)
- Warning Events: {warning_count} incidents in last 12 hours
- Equipment Temperature: {temp:.2f}°C average (normal range: 60-80°C)
- Vibration Level: {vibration:.2f} mm/s average (optimal: <2.0 mm/s)

Required Analysis Points:
1. Overall Equipment Effectiveness (OEE):
   - Analyze quality metrics impact on OEE
   - Evaluate current performance vs industry standards

2. Equipment Health Status:
   - Temperature and vibration trends
   - Warning events pattern analysis
   - Predictive maintenance recommendations

3. Actionable Recommendations:
   - Quality improvement steps
   - Maintenance scheduling
   - Energy efficiency opportunities

Please provide a thorough analysis with specific, actionable insights based on the data."""

        formatted_prompt = analysis_prompt.format(
            quality_score=metrics_data["avg_value"],
            warning_count=warning_count,
            temp=temp_response["result"]["statistics"]["avg_value"],
            vibration=vibration_response["result"]["statistics"]["avg_value"]
        )
        
        # Get LLM analysis
        llm_payload = {
            "jsonrpc": "2.0",
            "method": "mcp.get_information",
            "params": {
                "prompt": formatted_prompt
            },
            "id": "analysis-1"
        }
        
        print("\nSending analysis request to LLM...")
        llm_response = requests.post(
            "http://localhost:8000/api/v1/mcp",
            json=llm_payload,
            headers={"Content-Type": "application/json"},
            timeout=60  # Increased timeout for longer responses
        )
        llm_response.raise_for_status()
        llm_data = llm_response.json()
        
        # Enhanced error handling with null safety
        if not llm_data:
            raise Exception("Empty response from server")
            
        # First check if we have a valid result
        if llm_data.get("result"):
            analysis_text = llm_data["result"].strip()
        # Then handle potential errors
        elif 'error' in llm_data:
            error = llm_data['error']
            if isinstance(error, dict):
                error_detail = error.get('data', {})
                error_msg = error_detail.get('message') or error.get('message', 'Unknown error')
            else:
                error_msg = str(error)
            raise Exception(f"LLM API Error: {error_msg}")
        else:
            raise Exception("Invalid response format: no result or error field")
            
        # Check if response appears truncated
        if len(analysis_text) > 0 and not analysis_text.strip().endswith((".", "!", "?")):
            print("\nWarning: LLM response appears to be truncated")
        
        print("\nManufacturing Analysis Results:")
        print("-" * 50)
        print("Time Series Metrics:")
        print(json.dumps({
            "quality_score": metrics_data,
            "warning_events": warning_count,
            "temperature": temp_response["result"]["statistics"],
            "vibration": vibration_response["result"]["statistics"]
        }, indent=2))
        print("\nLLM Analysis:")
        print(analysis_text)
        
    except requests.Timeout:
        print("Error: Request to MCP timed out")
    except requests.exceptions.RequestException as e:
        print(f"Network Error: {str(e)}")
    except KeyError as e:
        print(f"Response format error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    print("Setting up manufacturing database...")
    setup_manufacturing_db()
    print("Database setup complete!")
    
    print("\nRunning manufacturing analysis...")
    test_manufacturing_analysis()