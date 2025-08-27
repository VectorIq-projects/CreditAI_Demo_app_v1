import streamlit as st
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from typing import Optional, Dict, Any

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.is_connected = False
    
    async def connect(self):
        """Connect to the SEC Edgar MCP server"""
        try:
            # Server configuration
            server_params = StdioServerParameters(
                command="uv",
                args=["run", "sec-edgar-mcp"],
                env={
                    "SEC_EDGAR_USER_AGENT": "Your Name (your.email@domain.com)"
                }
            )
            
            # Create stdio client
            stdio_transport = stdio_client(server_params)
            
            # Initialize session
            self.session = ClientSession(stdio_transport[0], stdio_transport[1])
            
            # Initialize the session
            await self.session.initialize()
            
            self.is_connected = True
            return True
            
        except Exception as e:
            st.error(f"Failed to connect to MCP server: {str(e)}")
            return False
    
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.session:
            await self.session.close()
            self.is_connected = False
    
    async def list_tools(self):
        """List available tools from the MCP server"""
        if not self.session:
            return []
        
        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            st.error(f"Error listing tools: {str(e)}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call a specific tool with arguments"""
        if not self.session:
            return None
        
        try:
            response = await self.session.call_tool(tool_name, arguments)
            return response
        except Exception as e:
            st.error(f"Error calling tool {tool_name}: {str(e)}")
            return None

# Initialize session state
if 'mcp_client' not in st.session_state:
    st.session_state.mcp_client = MCPClient()

if 'tools' not in st.session_state:
    st.session_state.tools = []

# Streamlit App
st.title("SEC Edgar MCP Client")

# Connection status
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Connection Status")
    if st.session_state.mcp_client.is_connected:
        st.success("Connected to SEC Edgar MCP Server")
    else:
        st.error("Not connected to MCP Server")

with col2:
    if st.button("Connect" if not st.session_state.mcp_client.is_connected else "Disconnect"):
        if not st.session_state.mcp_client.is_connected:
            # Connect
            async def connect():
                success = await st.session_state.mcp_client.connect()
                if success:
                    tools = await st.session_state.mcp_client.list_tools()
                    st.session_state.tools = tools
            
            asyncio.run(connect())
            st.rerun()
        else:
            # Disconnect
            async def disconnect():
                await st.session_state.mcp_client.disconnect()
            
            asyncio.run(disconnect())
            st.session_state.tools = []
            st.rerun()

# Show available tools
if st.session_state.mcp_client.is_connected and st.session_state.tools:
    st.subheader("Available Tools")
    
    for tool in st.session_state.tools:
        with st.expander(f"🔧 {tool.name}"):
            st.write(f"**Description:** {tool.description}")
            
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                st.write("**Parameters:**")
                if 'properties' in tool.inputSchema:
                    for param_name, param_info in tool.inputSchema['properties'].items():
                        param_type = param_info.get('type', 'unknown')
                        param_desc = param_info.get('description', 'No description')
                        required = param_name in tool.inputSchema.get('required', [])
                        req_text = " (Required)" if required else " (Optional)"
                        st.write(f"- **{param_name}** ({param_type}){req_text}: {param_desc}")

# Tool interaction section
if st.session_state.mcp_client.is_connected and st.session_state.tools:
    st.subheader("Use Tools")
    
    # Tool selection
    tool_names = [tool.name for tool in st.session_state.tools]
    selected_tool = st.selectbox("Select a tool to use:", tool_names)
    
    if selected_tool:
        # Find the selected tool
        tool = next((t for t in st.session_state.tools if t.name == selected_tool), None)
        
        if tool and hasattr(tool, 'inputSchema') and tool.inputSchema:
            # Dynamic form generation based on tool schema
            arguments = {}
            
            if 'properties' in tool.inputSchema:
                st.write(f"**Configure {selected_tool}:**")
                
                for param_name, param_info in tool.inputSchema['properties'].items():
                    param_type = param_info.get('type', 'string')
                    param_desc = param_info.get('description', '')
                    required = param_name in tool.inputSchema.get('required', [])
                    
                    if param_type == 'string':
                        value = st.text_input(
                            f"{param_name}" + (" *" if required else ""),
                            help=param_desc,
                            key=f"{selected_tool}_{param_name}"
                        )
                        if value:
                            arguments[param_name] = value
                    
                    elif param_type == 'integer':
                        value = st.number_input(
                            f"{param_name}" + (" *" if required else ""),
                            help=param_desc,
                            key=f"{selected_tool}_{param_name}",
                            format="%d"
                        )
                        arguments[param_name] = int(value)
                    
                    elif param_type == 'boolean':
                        value = st.checkbox(
                            f"{param_name}" + (" *" if required else ""),
                            help=param_desc,
                            key=f"{selected_tool}_{param_name}"
                        )
                        arguments[param_name] = value
            
            # Execute tool
            if st.button(f"Execute {selected_tool}"):
                with st.spinner(f"Executing {selected_tool}..."):
                    async def execute_tool():
                        result = await st.session_state.mcp_client.call_tool(selected_tool, arguments)
                        return result
                    
                    result = asyncio.run(execute_tool())
                    
                    if result:
                        st.subheader("Results")
                        
                        # Display results based on content type
                        if hasattr(result, 'content') and result.content:
                            for content_item in result.content:
                                if hasattr(content_item, 'type'):
                                    if content_item.type == 'text':
                                        st.text(content_item.text)
                                    elif content_item.type == 'json':
                                        st.json(content_item.data)
                                    else:
                                        st.write(content_item)
                        else:
                            st.json(result.__dict__ if hasattr(result, '__dict__') else str(result))

# Configuration section
with st.sidebar:
    st.subheader("Configuration")
    
    # User agent configuration
    current_ua = os.environ.get('SEC_EDGAR_USER_AGENT', 'Your Name (your.email@domain.com)')
    new_ua = st.text_input("SEC Edgar User Agent:", value=current_ua)
    
    if st.button("Update User Agent"):
        os.environ['SEC_EDGAR_USER_AGENT'] = new_ua
        st.success("User Agent updated!")
    
    st.info("💡 Make sure to set your actual name and email in the User Agent field as required by SEC Edgar API.")
    
    # Connection info
    st.subheader("Server Info")
    st.code("""
Command: uv run sec-edgar-mcp
Transport: stdio
Environment: SEC_EDGAR_USER_AGENT
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and MCP (Model Context Protocol)")