import asyncio
import os
import json
import sys
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field  # Pydantic v2 导入
# 加载 .env 文件，确保 API Key 受到保护
load_dotenv()

class QuerySchema(BaseModel):  # 类名改为 PascalCase
    """
    定义查询的 Pydantic 模型
    """
    mcp_tool: str = Field(..., description="需要调用的mcp tool")
    mcp_input: Dict[str, Any] = Field(default_factory=dict, description="mcp tool的输入参数，必须是字典格式")

    # Pydantic v2 配置方式
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "mcp_tool": "csv_info",
                    "mcp_input": {}
                },
                {
                    "mcp_tool": "csv_plot", 
                    "mcp_input": {"col_name": "age"}
                }
            ]
        }
    }

class MCPClient:
    def __init__(self):
        """初始化 MCP 客户端"""
        self.exit_stack = AsyncExitStack()
        # 从环境变量读取配置
        self.openai_api_key = os.getenv("OPENAI_API_KEY")  # 读取 OpenAI API Key
        self.base_url = os.getenv("BASE_URL")  # 读取 BASE URL
        self.model = os.getenv("MODEL")  # 读取模型名称

        if not self.openai_api_key:
            raise ValueError("❌ 未找到 OpenAI API Key，请在 .env 文件中设置 OPENAI_API_KEY")

        # 初始化 OpenAI 客户端
        self.client = ChatOpenAI(
            api_key=self.openai_api_key, 
            base_url=self.base_url, 
            model=self.model, 
            temperature=0.7
        )
        self.session: Optional[ClientSession] = None
        self.parser = PydanticOutputParser(pydantic_object=QuerySchema)  # 使用新的类名
        self.prompt_template = PromptTemplate(
            template="根据用户的输入选择合适的MCP Tool.\n{format_instructions}\n{input}\n",
            input_variables=["input"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chat = self.prompt_template | self.client | self.parser

    async def connect_to_server(self, server_script_path: str):
        """连接到 MCP 服务器并列出可用工具"""
        # 根据脚本类型选择执行命令
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )

        # 启动 MCP 服务器并建立通信
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        # 列出 MCP 服务器上的工具
        response = await self.session.list_tools()
        tools = response.tools
        print("\n已连接到服务器，支持以下工具:", [tool.name for tool in tools])
        
        # 创建工具描述
        tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        
        # 更新 prompt template 包含可用工具信息
        self.prompt_template = PromptTemplate(
            template="""根据用户的输入选择合适的MCP Tool。

可用的工具:
{tools}

请根据用户查询选择最合适的工具。
- 如果工具不需要参数，mcp_input 应该是空字典 {{}}
- 如果工具需要参数，mcp_input 应该是包含参数的字典，例如 {{"col_name": "age"}}

{format_instructions}

用户输入: {input}
""",
            input_variables=["input"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
                "tools": tools_description
            }
        )
        
        self.chat = self.prompt_template | self.client | self.parser

    async def process_query(self, query: str):
        """
        使用大模型处理查询并调用可用的 MCP 工具 (Function Calling)
        """
        if not self.session:
            raise RuntimeError("❌ 还未连接到 MCP 服务器，请先调用 connect_to_server()")

        # 使用大模型处理查询
        response = await self.chat.ainvoke({"input": query})
        print(f"\n🤖 LLM 响应: {response}")

        # 调用 MCP 工具
        tool_name = response.mcp_tool
        tool_input = response.mcp_input

        if not tool_name:
            raise ValueError("❌ 未指定要调用的 MCP Tool")

        # 调用指定的 MCP 工具
        result = await self.session.call_tool(tool_name, arguments=tool_input)
        
        # 处理返回结果 - 正确解析 MCP 工具返回的数据
        if hasattr(result, 'content') and result.content:
            content = result.content[0]
            if hasattr(content, 'text'):
                # 解析 JSON 格式的返回结果
                try:
                    result_data = json.loads(content.text)
                    # 只返回 message 字段
                    return result_data.get("message", "工具执行完成")
                except json.JSONDecodeError:
                    return content.text
            else:
                return str(content)
        
        return "工具执行完成，但无返回内容"

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\n🤖 Persimmon MCP 客户端已启动！输入 'quit' 退出")
        while True:
            try:
                query = input("\n你: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)  # 发送用户输入到 OpenAI API
                print(f"\n🤖 LLM: {response}")
            except Exception as e:
                print(f"\n⚠️ 发生错误: {str(e)}")

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()


async def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
