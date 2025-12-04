import onnxruntime as ort
from transformers import BertTokenizerFast
import numpy as np
import os
from typing import List, Union, Any, Dict
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import asyncio
import time

from py.get_setting import DEFAULT_EBD_DIR 
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_PATH = os.path.join(DEFAULT_EBD_DIR, MODEL_NAME)

# ------------------------------------------------
# 1. MiniLM ONNX Predictor 类 (保持不变)
# ------------------------------------------------

class MiniLMOnnxPredictor:
    # ... (此处包含您提供的 MiniLMOnnxPredictor 类的完整代码) ...
    """
    MiniLM ONNX 模型预测器，用于生成词嵌入。
    （代码保持不变，为节省空间在此省略）
    """
    def __init__(self, model_dir: str, use_gpu: bool = False):
        self.model_dir = model_dir
        self.is_loaded = False
        
        if not self._check_files_exist():
            return

        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
            model_path_o4 = os.path.join(model_dir, "model_O4.onnx")
            model_path_generic = os.path.join(model_dir, "model.onnx")
            model_path = model_path_o4 if os.path.exists(model_path_o4) else model_path_generic
            
            if not os.path.exists(model_path):
                raise FileNotFoundError()
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_names = [i.name for i in self.session.get_inputs()]
            
            print(f"✅ MiniLM ONNX Predictor loaded from: {model_path}")
            self.is_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading MiniLM ONNX Predictor: {e}")
            self.is_loaded = False

    def _check_files_exist(self) -> bool:
        onnx_exists = os.path.exists(os.path.join(self.model_dir, "model_O4.onnx")) or \
                      os.path.exists(os.path.join(self.model_dir, "model.onnx"))
        tokenizer_exists = os.path.exists(os.path.join(self.model_dir, "tokenizer.json")) or \
                           os.path.exists(os.path.join(self.model_dir, "vocab.txt"))
        return onnx_exists and tokenizer_exists

    def mean_pooling(self, model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        token_embeddings = model_output
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(norm, a_min=1e-9, a_max=None)

    def predict(self, sentences: List[str]) -> np.ndarray:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Cannot run prediction.")
            
        inputs = self.tokenizer(
            sentences, padding=True, truncation=True, max_length=512, return_tensors="np" 
        )
        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
        if 'token_type_ids' in self.input_names:
            if 'token_type_ids' in inputs:
                ort_inputs['token_type_ids'] = inputs['token_type_ids'].astype(np.int64)
            else:
                ort_inputs['token_type_ids'] = np.zeros_like(inputs['input_ids'], dtype=np.int64)

        outputs = self.session.run(None, ort_inputs)
        last_hidden_state = outputs[0] 

        embeddings = self.mean_pooling(last_hidden_state, inputs['attention_mask'])
        embeddings = self.normalize(embeddings)

        return embeddings.astype(np.float32)

# ------------------------------------------------
# 2. FastAPI 模型和路由设置
# ------------------------------------------------

# 初始化全局预测器和路由
global_minilm_predictor = MiniLMOnnxPredictor(MODEL_PATH, use_gpu=False)
router = APIRouter(prefix="/minilm", tags=["MiniLM Embeddings (OpenAI Compatible)"])

# --- Pydantic 模型定义 ---

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = MODEL_NAME

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, Any]

# --- 依赖注入函数 ---

def get_minilm_predictor() -> MiniLMOnnxPredictor:
    """依赖注入函数，检查模型是否已加载"""
    if not global_minilm_predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{MODEL_NAME}' is not installed or failed to load. Please download it first."
        )
    return global_minilm_predictor

# --- 路由定义 ---

@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    predictor: MiniLMOnnxPredictor = Depends(get_minilm_predictor)
):
    """
    OpenAI 兼容的词嵌入接口。
    """
    start_time = time.time()
    
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input
        
    # 计算 Token 数 (近似值)
    num_tokens = sum(len(predictor.tokenizer.tokenize(t)) for t in texts)
    
    # 在后台线程中运行同步的 ONNX 推理
    try:
        embeddings_np = await asyncio.to_thread(predictor.predict, texts)
        
    except Exception as e:
        print(f"MiniLM Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # 格式化响应数据
    response_data = [
        EmbeddingData(embedding=emb.tolist(), index=i) 
        for i, emb in enumerate(embeddings_np)
    ]
        
    return EmbeddingResponse(
        model=request.model,
        data=response_data,
        usage={
            "prompt_tokens": num_tokens,
            "total_tokens": num_tokens,
            "inference_time_ms": int((time.time() - start_time) * 1000)
        }
    )