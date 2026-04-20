from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import tool

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import os
import asyncio
from contextlib import asynccontextmanager

# ---------------- ENV ----------------
load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# ---------------- GLOBALS ----------------
mcp_session: ClientSession | None = None
mcp_session_context = None
mcp_stdio_context = None

# ---------------- LIFESPAN ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles both startup and shutdown of the MCP connection."""
    global mcp_session, mcp_session_context, mcp_stdio_context

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]  # path to your MCP server file
    )

    try:
        # Open stdio connection to MCP server
        mcp_stdio_context = stdio_client(server_params)
        read, write = await mcp_stdio_context.__aenter__()

        # Create and initialize the MCP session
        mcp_session_context = ClientSession(read, write)
        mcp_session = await mcp_session_context.__aenter__()
        await mcp_session.initialize()

        tools = await mcp_session.list_tools()
        print(f"✅ MCP Connected | Tools available: {[t.name for t in tools.tools]}")

    except Exception as e:
        print(f"❌ MCP Connection failed: {e}")
        mcp_session = None

    yield  # Application runs here

    # Shutdown: clean up MCP connection
    print("🔄 Shutting down MCP connection...")
    if mcp_session_context:
        await mcp_session_context.__aexit__(None, None, None)
    if mcp_stdio_context:
        await mcp_stdio_context.__aexit__(None, None, None)
    print("✅ MCP disconnected cleanly")


# ---------------- APP ----------------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


class RequestPayload(BaseModel):
    message: str


# ---------------- CUSTOM TOOLS ----------------
# Using @tool decorator so LangChain agent can properly detect and call these tools.
# Tool name must match exactly what the agent sees — avoid spaces, use underscores.

@tool
def AI_name(query: str) -> str:
    """Returns the name of this AI assistant. Use this when asked 'what is your name' or 'who are you'."""
    return "I am AI Assistant built using MCP tools and LangChain."

@tool
def AI_version(query: str) -> str:
    """Returns the current version of this AI assistant. Use this when asked about the version."""
    return "Current version is 1.0.0"

@tool
def AI_description(query: str) -> str:
    """Returns a description of what this AI assistant can do. Use this when asked 'what can you do' or 'describe yourself'."""
    return "I am an AI assistant that can perform arithmetic operations (add, subtract, multiply, divide) using MCP tools, and also answer questions about myself."

# Collect all custom tools in a list
CUSTOM_TOOLS = [AI_name, AI_version, AI_description]


# ---------------- MCP TOOL LOADER ----------------

async def load_mcp_tools() -> list[Tool]:
    """Dynamically loads tools from the connected MCP server."""
    global mcp_session

    if mcp_session is None:
        print("⚠️  MCP session is not available")
        return []

    tools = []

    try:
        mcp_tools_result = await mcp_session.list_tools()

        for t in mcp_tools_result.tools:
            tool_name = t.name
            tool_desc = (
                (t.description or "No description")
                + " | Input format: 'number1 number2' (e.g. '10 5')"
            )

            # Async function to call the MCP tool — captures tool_name correctly via default arg
            async def call_mcp_tool(input_str: str, _tool_name: str = tool_name) -> str:
                try:
                    parts = input_str.strip().split()
                    if len(parts) < 2:
                        return "Error: Two numbers are required. Example input: '10 5'"

                    args = {
                        "a": float(parts[0]),
                        "b": float(parts[1])
                    }

                    result = await mcp_session.call_tool(_tool_name, args)

                    # Extract text from result content
                    if result.content and len(result.content) > 0:
                        return result.content[0].text
                    return str(result)

                except ValueError:
                    return "Error: Please provide valid numbers. Example: '10 5'"
                except Exception as e:
                    return f"Error calling {_tool_name}: {str(e)}"

            # Sync wrapper using current event loop (avoids nested asyncio.run() issues)
            def make_sync_tool(async_fn):
                def sync_fn(x: str) -> str:
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(async_fn(x))
                return sync_fn

            tools.append(
                Tool(
                    name=tool_name,
                    func=make_sync_tool(call_mcp_tool),
                    description=tool_desc,
                    coroutine=call_mcp_tool  # async version for ainvoke compatibility
                )
            )

        print(f"🔧 MCP tools loaded: {[t.name for t in tools]}")

    except Exception as e:
        print(f"❌ Error loading MCP tools: {e}")

    return tools


# ---------------- CHAT API ----------------

@app.post("/chat")
async def chat(request: RequestPayload):
    user_input = request.message
    print(f"📩 User input received: {user_input}")

    try:
        # Load tools from MCP server
        mcp_tools = await load_mcp_tools()

        # Merge custom tools + MCP tools
        all_tools = CUSTOM_TOOLS + mcp_tools

        print(f"🛠️  All tools available to agent: {[t.name for t in all_tools]}")

        # Pull the standard ReAct prompt from LangChain Hub (required by create_react_agent)
        prompt = hub.pull("hwchase17/react")

        # Create the ReAct agent with all tools
        agent = create_react_agent(llm, all_tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=all_tools,
            verbose=True,
            handle_parsing_errors=True,  # Gracefully handle LLM output parsing errors
            max_iterations=5             # Prevent infinite reasoning loops
        )

        result = await agent_executor.ainvoke({"input": user_input})

        return {
            "response": result["output"],
            "status": "success"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}


# ---------------- ROOT ----------------

@app.get("/")
def read_root():
    return {
        "message": "✅ MCP + LLM API is running",
        "mcp_connected": mcp_session is not None
    }


# ---------------- RUN ----------------

if __name__ == "__main__":
    import uvicorn
    # reload=False is important — reload restarts lifespan and breaks MCP session
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False)