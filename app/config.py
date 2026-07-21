"""Cấu hình ứng dụng — đọc từ biến môi trường / file .env.

Đây là điểm DUY NHẤT đọc secrets. Phần còn lại của code import `settings`
chứ không gọi os.environ trực tiếp → dễ test, dễ kiểm soát.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── Backend (Buổi 2: Local Serving) ──────────────────────────────────────
    # "openai" = OpenAI Cloud | "ollama" = Ollama local | "vllm" = vLLM server.
    # Cả 3 đều nói chuyện qua OpenAI-compatible API → phần còn lại của code không đổi.
    llm_backend: str = Field(default="openai", alias="LLM_BACKEND")
    # base_url override. Để trống → dùng mặc định của backend (xem llm/backends.py).
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")

    # ── OpenAI ──────────────────────────────────────────────────────────────
    # Một hoặc nhiều API key, ngăn cách bằng dấu phẩy (bật key rotation).
    # Với backend local (ollama/vllm) không cần key thật.
    openai_api_keys: str = Field(default="", alias="OPENAI_API_KEYS")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    # ── Generation params (mặc định) ────────────────────────────────────────
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_max_completion_tokens: int = Field(default=800, alias="LLM_MAX_COMPLETION_TOKENS")
    llm_top_p: float = Field(default=1.0, alias="LLM_TOP_P")

    # ── Resilience ──────────────────────────────────────────────────────────
    llm_max_retries: int = Field(default=5, alias="LLM_MAX_RETRIES")

    # ── Embeddings & Vector Store (Buổi 4) ───────────────────────────────────
    # OpenAI text-embedding-3-small: 1536 dims, gọi API (dùng chung key với LLM).
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")

    # Qdrant. Mặc định ":memory:" cho demo (không cần Docker). Production: đặt
    #   QDRANT_URL=http://localhost:6333  và chạy Qdrant qua Docker.
    qdrant_url: str = Field(default=":memory:", alias="QDRANT_URL")
    vectorstore_collection: str = Field(
        default="legal_docs", alias="VECTORSTORE_COLLECTION"
    )

    # ── RAG Pipeline (Buổi 5) ────────────────────────────────────────────────
    rag_source_dir: str = Field(default="./data/legal_docs", alias="RAG_SOURCE_DIR")
    rag_chunk_size: int = Field(default=512, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=64, alias="RAG_CHUNK_OVERLAP")
    # Contextual Retrieval: prepend heading path vào chunk markdown (cải thiện retrieval).
    rag_contextual_chunking: bool = Field(default=True, alias="RAG_CONTEXTUAL_CHUNKING")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")

    # Query rewriting: LLM viết lại câu hỏi trước khi search (cải thiện recall).
    rag_query_rewriting: bool = Field(default=False, alias="RAG_QUERY_REWRITING")

    # Re-ranking: cross-encoder lọc lại sau retrieve (cần sentence-transformers).
    rag_rerank_enabled: bool = Field(default=False, alias="RAG_RERANK_ENABLED")
    rag_fetch_k: int = Field(default=20, alias="RAG_FETCH_K")  # lấy rộng trước khi rerank
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL"
    )

    # ── Agentic RAG / CRAG (Buổi 6) ──────────────────────────────────────────
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    # Query decomposition: chỉ decompose câu hỏi đủ dài/phức tạp (heuristic đơn giản).
    agent_decompose_min_chars: int = Field(default=80, alias="AGENT_DECOMPOSE_MIN_CHARS")
    agent_max_sub_questions: int = Field(default=4, alias="AGENT_MAX_SUB_QUESTIONS")
    # Số chunk "relevant" tối thiểu để KHÔNG cần web search fallback.
    agent_min_relevant_chunks: int = Field(default=1, alias="AGENT_MIN_RELEVANT_CHUNKS")
    agent_web_search_results: int = Field(default=3, alias="AGENT_WEB_SEARCH_RESULTS")

    # ── App ─────────────────────────────────────────────────────────────────
    app_name: str = Field(default="Vietnamese Legal Assistant", alias="APP_NAME")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def api_keys(self) -> list[str]:
        """Tách chuỗi key thành list, bỏ khoảng trắng và phần tử rỗng."""
        return [k.strip() for k in self.openai_api_keys.split(",") if k.strip()]

    @field_validator("openai_api_keys")
    @classmethod
    def _warn_if_empty(cls, v: str) -> str:
        # Không raise ở đây để test/import không cần key thật;
        # llm/client.py sẽ báo lỗi rõ ràng khi thực sự cần gọi API.
        return v


@lru_cache
def get_settings() -> Settings:
    """Singleton settings (cache để không đọc .env nhiều lần)."""
    return Settings()


settings = get_settings()
