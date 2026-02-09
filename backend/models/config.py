from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from enum import Enum


class Modality(str, Enum):
    """Supported modalities"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABLE = "table"
    PDF = "pdf"
    DOCUMENT = "document"


class Resolution(str, Enum):
    """Resolution levels for knowledge atoms"""
    FINE = "fine"
    MID = "mid"
    COARSE = "coarse"


class EdgeType(str, Enum):
    """Graph edge types"""
    PART_OF = "PART_OF"
    NEXT = "NEXT"
    DERIVED_FROM = "DERIVED_FROM"
    SIMILAR_TO = "SIMILAR_TO"
    REFERENCES = "REFERENCES"
    ENTITY_SHARED = "ENTITY_SHARED"
    ALIGNED_WITH = "ALIGNED_WITH"
    VISUAL_GROUNDED_IN = "VISUAL_GROUNDED_IN"
    SCENE_DESCRIBES = "SCENE_DESCRIBES"


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    GROQ = "groq"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class APIKeyInstance(BaseModel):
    """Individual API key instance"""
    name: str
    key: str
    provider: LLMProvider
    model_name: Optional[str] = None
    active: bool = True
    request_count: int = 0
    error_count: int = 0
    last_used: Optional[float] = None


class LLMConfig(BaseModel):
    """LLM Engine configuration"""
    instances: Dict[LLMProvider, List[APIKeyInstance]] = Field(default_factory=dict)
    preferred_provider: Optional[LLMProvider] = None
    token_budget: int = 8000
    temperature: float = 0.7
    max_retries: int = 3


class RetrievalWeights(BaseModel):
    """Adaptive retrieval weights (α, β, γ, δ, ε)"""
    alpha: float = Field(default=0.3, ge=0.0, le=1.0)  # Vector similarity
    beta: float = Field(default=0.25, ge=0.0, le=1.0)  # Graph proximity
    gamma: float = Field(default=0.2, ge=0.0, le=1.0)  # BM25
    delta: float = Field(default=0.15, ge=0.0, le=1.0)  # Structural importance
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)  # Modality alignment

    def normalize(self):
        """Ensure weights sum to 1"""
        total = self.alpha + self.beta + self.gamma + self.delta + self.epsilon
        if total > 0:
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
            self.delta /= total
            self.epsilon /= total


class KnowledgeAtom(BaseModel):
    """Multi-Resolution Semantic Unit"""
    content: str
    modality: Modality
    resolution: Resolution
    embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    metadata: Dict = Field(default_factory=dict)
    source: str
    timestamp: float
    node_id: Optional[str] = None
    entities: List[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    """Graph edge definition"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = Field(ge=0.0, le=1.0)
    metadata: Dict = Field(default_factory=dict)


class QueryDecomposition(BaseModel):
    """Decomposed query structure"""
    sub_queries: List[str]
    modalities: List[Modality]
    intent_weights: List[float]
    depth: int = 2
    freshness_required: bool = False
    reasoning_required: bool = True


class RetrievalResult(BaseModel):
    """Single retrieval result"""
    atom_id: str
    content: str
    modality: Modality
    score: float
    vector_score: float = 0.0
    graph_score: float = 0.0
    bm25_score: float = 0.0
    struct_score: float = 0.0
    mod_score: float = 0.0
    context: List[str] = Field(default_factory=list)
    source: Optional[str] = None


class SearchConfig(BaseModel):
    """Search configuration"""
    top_k_vector: int = 5
    top_k_graph: int = 5
    top_k_bm25: int = 3
    graph_hops: int = 2
    pagerank_alpha: float = 0.15
    diversity_lambda: float = 0.3
    confidence_threshold: float = 0.6


class PerformanceMetrics(BaseModel):
    """Performance tracking metrics"""
    hit_rate: float = 0.0
    semantic_overlap: float = 0.0
    mmr_score: float = 0.0
    user_engagement: float = 0.0
    timestamp: float = 0.0


class DatabaseConfig(BaseModel):
    """ArangoDB configuration"""
    host: str = "localhost"
    port: int = 8529
    username: str = "root"
    password: str = "rootpassword"
    database: str = "multimodal_rag"
    
    # Collections
    nodes_collection: str = "knowledge_atoms"
    edges_collection: str = "semantic_edges"
    
    # Graph
    graph_name: str = "knowledge_graph"