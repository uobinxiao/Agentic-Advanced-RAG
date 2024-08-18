from Utils import *
from Frontend import *
from typing import Optional
from dataclasses import dataclass
from .constants import MODEL_NAME, MODEL_TEMPERATURE

@dataclass
class RAGConfig:
    model_name: str = MODEL_NAME
    model_temperature: float = MODEL_TEMPERATURE
    vector_database: Optional['VectorDatabase'] = None
    embedder: Optional['Embedder'] = None
    callback_function: Optional['CustomStreamlitCallbackHandler'] = None