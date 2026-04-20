from mcp.server.fastmcp import FastMCP

mcp = FastMCP('Calculator Server')

@mcp.tool()
def add(a, b):
    """Adds two numbers."""
    return f" Result of two number is {a + b}"


@mcp.tool()
def subtract(a, b):
    """Subtracts the second number from the first."""
    return f" Result of two number is {a - b}"


@mcp.tool()
def multiply(a, b):
    """Multiplies two numbers."""
    return f" Result of two number is {a * b}"


@mcp.tool()
def divide(a, b):
    if b == 0:
        return "Cannot divide by zero"
    
    return f"Result of two number is {a / b}"


if __name__ == "__main__":
    mcp.run()