import asyncio
import httpx # 核心修复：使用异步 HTTP 客户端
from typing import List, Dict, Union
import json
import os
from pathlib import Path
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from py.load_files import get_files_json
from py.get_setting import load_settings, base_path, KB_DIR
    
# --- Tiktoken 缓存设置（保留）---
def get_tiktoken_cache_path():
    cache_path = os.path.join(base_path, "tiktoken_cache")
    os.makedirs(cache_path, exist_ok=True)
    return cache_path

os.environ["TIKTOKEN_CACHE_DIR"] = get_tiktoken_cache_path()
# ---------------------------------


class MyOpenAICompatibleEmbeddings(Embeddings):
    """
    OpenAI 兼容的词嵌入类，使用 httpx 异步客户端进行非阻塞网络请求。
    """
    def __init__(self, base_url: str, model: str, api_key: str = "empty"):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        # 假设 base_url 已经是 http://127.0.0.1:8000/minilm
        self.endpoint = f"{self.base_url}/embeddings"

    # --- 异步核心方法 ---
    async def _aembed(self, texts: Union[str, List[str]]) -> List[Dict]:
        """异步发送嵌入请求并处理响应"""
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        json_data = {"model": self.model, "input": texts}
        
        # 使用 httpx.AsyncClient 发送请求
        # timeout=None 是为了避免在大型嵌入任务中请求超时
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                # 调用词嵌入接口
                response = await client.post(self.endpoint, headers=headers, json=json_data)
                
                # 检查 HTTP 状态码，如果不是 2xx 则抛出异常
                response.raise_for_status() 
                
                return response.json()["data"]
                
            except httpx.HTTPStatusError as e:
                # 捕捉 HTTP 状态码错误 (例如 503 模型未加载)
                detail = e.response.json().get('detail', e.response.text) if e.response.text else 'Unknown error'
                raise RuntimeError(f"Embedding API HTTP Error {e.response.status_code}: {detail}")
            except Exception as e:
                # 捕捉连接错误
                raise ConnectionError(f"Embedding API connection failed: {e.__class__.__name__}: {e}")

    # --- LangChain 兼容的同步方法 ---
    # 由于 LangChain 在构建向量库时会同步调用这些方法，我们必须在同步方法中运行异步核心。
    
    def embed_query(self, text: str) -> List[float]:
        # 在同步方法中运行异步任务
        # 注意: 这种做法可能在 LangChain 内部的多线程环境中有风险，但可以解决死锁问题。
        data = asyncio.run(self.aembed_query(text))
        return data

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 在同步方法中运行异步任务
        data = asyncio.run(self.aembed_documents(texts))
        return data

    # --- 暴露异步 LangChain 方法 (供内部调用) ---
    # LangChain 的新版本倾向于使用这些异步方法。如果您的 LangChain 版本支持，可以直接调用它们。
    
    async def aembed_query(self, text: str) -> List[float]:
        data = await self._aembed(text)
        return data[0]["embedding"]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        data = await self._aembed(texts)
        return [r["embedding"] for r in data]


def chunk_documents(results: List[Dict], cur_kb) -> List[Document]:
    """为每个文件单独分块并添加元数据"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cur_kb["chunk_size"],
        chunk_overlap=cur_kb["chunk_overlap"],
        separators=["\n\n", "\n", "。", "！", "？", "!", "?", "."]
    )
    
    all_docs = []
    for doc in results:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            all_docs.append(Document(
                page_content=chunk,
                metadata={
                    "file_path": doc["file_path"],
                    "file_name": doc["file_name"],
                    "doc_id": f"{doc['file_path']}_{len(all_docs)}" 
                }
            ))
    return all_docs

# 核心修改：将 build_vector_store 改为异步函数，以便在内部调用 MyOpenAICompatibleEmbeddings 的异步方法
async def build_vector_store(docs: List[Document], kb_id, cur_kb: Dict, cur_vendor: str):
    """构建并保存双索引（参数修正版）"""
    if not isinstance(docs, list) or not all(isinstance(d, Document) for d in docs):
        raise ValueError("Input must be a list of Document objects")
    
    # ========== BM25索引构建 (同步操作，但放在异步函数中，避免阻塞) ==========
    try:
        kb_dir = Path(KB_DIR)
        kb_dir.mkdir(parents=True, exist_ok=True)
        save_dir = kb_dir / str(kb_id)
        save_dir.mkdir(parents=True, exist_ok=True)

        bm25_path = save_dir / "bm25_index.json"
        
        if not docs:
            raise ValueError("Documents list is empty")
            
        # 使用 asyncio.to_thread 确保同步文件I/O不会阻塞事件循环
        await asyncio.to_thread(
            lambda: json.dump({
                "docs": [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in docs
                ]
            }, open(bm25_path, "w", encoding="utf-8"), ensure_ascii=False)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save BM25 index: {str(e)}")
        
    # ========== 向量索引构建 (使用异步客户端) ==========
    try:
        embeddings = MyOpenAICompatibleEmbeddings(
            model=cur_kb["model"],
            api_key=cur_kb["api_key"],
            base_url=cur_kb["base_url"],
        )
        
        # 批量处理文档，LangChain 的 FAISS.from_documents 和 add_documents 会调用 
        # embeddings.embed_documents，该方法内部已通过 asyncio.run 适配
        batch_size = 5 
        vector_db = None
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            
            # 由于 LangChain 的 FAISS 构造器是同步的，我们使用 asyncio.to_thread 来运行它
            if vector_db is None:
                vector_db = await asyncio.to_thread(FAISS.from_documents, batch, embeddings)
            else:
                await asyncio.to_thread(vector_db.add_documents, batch)
            
            print(f"Processed {min(i+batch_size, len(docs))}/{len(docs)} documents")
        
        # 最终保存
        save_path = Path(KB_DIR) / str(kb_id)
        # 确保保存操作也在线程中完成
        await asyncio.to_thread(vector_db.save_local, folder_path=str(save_path), index_name="index")
        
    except Exception as e:
        # 如果模型未加载，这里会捕获到 RuntimeError/ConnectionError
        raise RuntimeError(f"Vector store build failed: {str(e)}")


async def load_retrievers(kb_id, cur_kb, cur_vendor):
    """加载双检索器"""
    # 加载BM25
    bm25_path = Path(KB_DIR) / str(kb_id) / "bm25_index.json"
    
    # 异步读取文件
    bm25_data = await asyncio.to_thread(json.load, open(bm25_path, "r", encoding="utf-8"))
    
    bm25_docs = [
        Document(
            page_content=doc["page_content"],
            metadata=doc["metadata"]
        ) for doc in bm25_data["docs"]
    ]
    # BM25Retriever 构造器是同步的
    bm25_retriever = await asyncio.to_thread(BM25Retriever.from_documents, bm25_docs)
    bm25_retriever.k = cur_kb["chunk_k"]
    
    # 加载向量检索器
    kb_path = Path(KB_DIR) / str(kb_id)
    embeddings = MyOpenAICompatibleEmbeddings(
        model=cur_kb["model"],
        api_key=cur_kb["api_key"],
        base_url=cur_kb["base_url"],
    )
    
    # FAISS.load_local 是同步的，内部会调用 embed_query
    vector_db = await asyncio.to_thread(
        FAISS.load_local,
        folder_path=str(kb_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
        index_name="index"
    )
    vector_retriever = vector_db.as_retriever(
        search_kwargs={"k": cur_kb["chunk_k"]}
    )
    return bm25_retriever, vector_retriever

async def query_vector_store(query: str, kb_id, cur_kb, cur_vendor):
    """使用EnsembleRetriever的混合查询"""
    bm25_retriever, vector_retriever = await load_retrievers(kb_id, cur_kb, cur_vendor)
    if "weight" not in cur_kb:
        cur_kb["weight"] = 0.5
        
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[1 - cur_kb["weight"], cur_kb["weight"]],
    )
    
    # EnsembleRetriever.invoke 是同步阻塞的，需要放在线程中运行
    docs = await asyncio.to_thread(ensemble_retriever.invoke, query)
    
    # 格式转换
    return [{
        "content": doc.page_content,
        "metadata": doc.metadata,
    } for doc in docs]


async def process_knowledge_base(kb_id):
    """异步处理知识库的完整流程"""
    settings = await load_settings()
    cur_kb = None
    providerId = None
    for kb in settings["knowledgeBases"]:
        if kb["id"] == kb_id:
            cur_kb = kb
            providerId = kb["providerId"]
            break
    cur_vendor = None
    for provider in settings["modelProviders"]:
        if provider["id"] == providerId:
            cur_vendor = provider["vendor"]
            break
    
    if not cur_kb:
        raise ValueError(f"Knowledge base {kb_id} not found in settings")
        
    processed_results = await get_files_json(cur_kb["files"])
    
    chunks = chunk_documents(processed_results, cur_kb)
    
    # 调用异步版本的 build_vector_store
    await build_vector_store(chunks, kb_id, cur_kb, cur_vendor)

    return "知识库处理完成"

async def query_knowledge_base(kb_id, query: str):
    """查询知识库"""
    settings = await load_settings()
    cur_kb = None
    providerId = None
    for kb in settings["knowledgeBases"]:
        if kb["id"] == kb_id:
            cur_kb = kb
            providerId = kb["providerId"]
            break
    cur_vendor = None
    for provider in settings["modelProviders"]:
        if provider["id"] == providerId:
            cur_vendor = provider["vendor"]
            break
    
    if not cur_kb:
        return f"Knowledge base {kb_id} not found in settings"
        
    # 调用异步版本的 query_vector_store
    results = await query_vector_store(query, kb_id, cur_kb, cur_vendor)
    return results

async def rerank_knowledge_base(query: str , docs: List[Dict]) -> List[Dict]:
    settings = await load_settings()
    providerId = settings["KBSettings"]["selectedProvider"]
    cur_vendor = None
    for provider in settings["modelProviders"]:
        if provider["id"] == providerId:
            cur_vendor = provider["vendor"]
            break
    if cur_vendor == "jina":
        jina_api_key = settings["KBSettings"]["api_key"]
        model_name = settings["KBSettings"]["model"]
        top_n = settings["KBSettings"]["top_n"]
        documents = [doc.get("content", "") for doc in docs]
        url = settings["KBSettings"]["base_url"] + "/rerank"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jina_api_key}"
        }
        data = {
            "model": model_name,
            "query": query,
            "top_n": top_n,
            "documents": documents,
            "return_documents": False
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Jina reranking failed: {response.text}")
        result = response.json()
        ranked_indices = [item['index'] for item in result.get('results', [])]
        ranked_docs = [docs[i] for i in ranked_indices]
        return ranked_docs
    elif cur_vendor == "Vllm":
        model_name = settings["KBSettings"]["model"]
        top_n = settings["KBSettings"]["top_n"]
        documents = [doc.get("content", "") for doc in docs]
        url = settings["KBSettings"]["base_url"] + "/rerank"
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        data = {
            "model": model_name,
            "query": query,
            "top_n": top_n,
            "documents": documents,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Vllm reranking failed: {response.text}")
        result = response.json()
        ranked_indices = [item['index'] for item in result.get('results', [])]
        ranked_docs = [docs[i] for i in ranked_indices]
        return ranked_docs
    else:
        return docs

kb_tool = {
    "type": "function",
    "function": {
        "name": "query_knowledge_base",
        "description": f"通过自然语言获取的对应ID的知识库信息。回答时，在回答的最下方给出信息来源。以链接的形式给出信息来源，格式为：[file_name](file_path)。file_path可以是外部资源，也可以是127.0.0.1上的资源。返回链接时，不要让()内出现空格",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "需要搜索的问题。",
                },
                "kb_id": {
                    "type": "string",
                    "description": "知识库的ID。"
                }
            },
            "required": ["kb_id","query"],
        },
    },
}