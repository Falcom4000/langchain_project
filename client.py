import asyncio
import os
import json
import sys
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pydantic import BaseModel, Field
import pandas as pd

# 加载 .env 文件，确保 API Key 受到保护
load_dotenv()

class QuerySchema(BaseModel):
    """
    定义查询的 Pydantic 模型
    """
    mcp_tool: str = Field(..., description="需要调用的mcp tool")
    mcp_input: Dict[str, Any] = Field(default_factory=dict, description="mcp tool的输入参数，必须是字典格式")

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

class RAGManager:
    """RAG知识库管理器"""
    
    def __init__(self, openai_api_key: str, base_url: Optional[str] = None):
        self.embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,
            base_url=base_url
        )
        self.vectorstore: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    async def load_documents(self, doc_paths: List[str]) -> List[Document]:
        """加载多种类型的文档"""
        documents = []
        
        for path in doc_paths:
            if not os.path.exists(path):
                print(f"⚠️ 文件不存在: {path}")
                continue
                
            file_ext = os.path.splitext(path)[1].lower()
            
            try:
                if file_ext == '.csv':
                    # 加载CSV文件
                    df = pd.read_csv(path)
                    # 创建CSV概览文档
                    content = f"CSV文件概览 - {os.path.basename(path)}:\n"
                    content += f"形状: {df.shape}\n"
                    content += f"列名: {', '.join(df.columns.tolist())}\n"
                    content += f"数据类型:\n{df.dtypes.to_string()}\n"
                    content += f"前5行数据:\n{df.head().to_string()}\n"
                    content += f"统计信息:\n{df.describe().to_string()}"
                    
                    doc = Document(
                        page_content=content,
                        metadata={"source": path, "type": "csv"}
                    )
                    documents.append(doc)
                    
                elif file_ext in ['.txt', '.md']:
                    # 加载文本文件
                    loader = TextLoader(path)
                    docs = loader.load()
                    documents.extend(docs)
                    
                print(f"✅ 已加载文档: {path}")
                
            except Exception as e:
                print(f"❌ 加载文档失败 {path}: {e}")
        
        return documents
    
    async def build_vectorstore(self, documents: List[Document]):
        """构建向量数据库"""
        if not documents:
            print("⚠️ 没有文档用于构建向量数据库")
            return
            
        # 分割文档
        texts = self.text_splitter.split_documents(documents)
        print(f"📄 文档分割完成，共 {len(texts)} 个片段")
        
        # 构建向量数据库
        self.vectorstore = await FAISS.afrom_documents(texts, self.embeddings)
        print("✅ 向量数据库构建完成")
    
    async def retrieve_context(self, query: str, k: int = 3) -> str:
        """检索相关上下文"""
        if not self.vectorstore:
            return ""
            
        try:
            docs = await self.vectorstore.asimilarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            print(f"❌ 检索上下文失败: {e}")
            return ""

class MCPClient:
    def __init__(self):
        """初始化 MCP 客户端"""
        self.exit_stack = AsyncExitStack()
        # 从环境变量读取配置
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")

        if not self.openai_api_key:
            raise ValueError("❌ 未找到 OpenAI API Key，请在 .env 文件中设置 OPENAI_API_KEY")

        # 初始化 OpenAI 客户端
        self.client = ChatOpenAI(
            api_key=self.openai_api_key, 
            base_url=self.base_url, 
            model=self.model, 
            temperature=0.7
        )
        
        # 初始化 RAG 管理器
        self.rag_manager = RAGManager(self.openai_api_key, self.base_url)
        
        self.session: Optional[ClientSession] = None
        self.parser = PydanticOutputParser(pydantic_object=QuerySchema)
        self.prompt_template = PromptTemplate(
            template="根据用户的输入选择合适的MCP Tool.\n{format_instructions}\n{input}\n",
            input_variables=["input"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chat = self.prompt_template | self.client | self.parser

    async def initialize_rag(self, knowledge_base_paths: List[str]):
        """初始化RAG知识库"""
        print("🔄 正在初始化RAG知识库...")
        documents = await self.rag_manager.load_documents(knowledge_base_paths)
        await self.rag_manager.build_vectorstore(documents)
        print("✅ RAG知识库初始化完成")

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
        
        tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        
        # 更新prompt template以支持RAG上下文
        self.prompt_template = PromptTemplate(
            template="""根据用户的输入和相关上下文信息选择合适的MCP Tool。

可用的工具:
{tools}

相关上下文信息:
{context}

请根据用户查询和上下文信息选择最合适的工具。
- 如果工具不需要参数，mcp_input 应该是空字典 {{}}
- 如果工具需要参数，mcp_input 应该是包含参数的字典，例如 {{"col_name": "age"}}

{format_instructions}

用户输入: {input}
""",
            input_variables=["input", "context"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
                "tools": tools_description
            }
        )
        
        self.chat = self.prompt_template | self.client | self.parser

    async def process_query(self, query: str):
        """
        使用RAG增强的查询处理
        """
        if not self.session:
            raise RuntimeError("❌ 还未连接到 MCP 服务器，请先调用 connect_to_server()")

        # RAG检索相关上下文
        context = await self.rag_manager.retrieve_context(query)
        if context:
            print(f"🔍 检索到相关上下文 ({len(context)} 字符)")
        
        # 使用大模型处理查询（包含上下文）
        response = await self.chat.ainvoke({"input": query, "context": context})
        print(f"\n🤖 LLM 响应: {response}")

        # 调用 MCP 工具
        tool_name = response.mcp_tool
        tool_input = response.mcp_input

        if not tool_name:
            raise ValueError("❌ 未指定要调用的 MCP Tool")

        result = await self.session.call_tool(tool_name, arguments=tool_input)
        
        return result.content[0].text

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
        print("Usage: python client.py <path_to_server_script> [knowledge_base_paths...]")
        print("Example: python client.py server.py data/titanic_cleaned.csv docs/readme.txt")
        sys.exit(1)

    client = MCPClient()
    try:
        # 初始化RAG知识库（如果提供了知识库路径）
        if len(sys.argv) > 2:
            knowledge_paths = sys.argv[2:]
            await client.initialize_rag(knowledge_paths)
        
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
