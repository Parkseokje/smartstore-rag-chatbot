import chromadb
from chromadb.utils import embedding_functions
import logging


class ChromaDBService:
    """ChromaDB 관련 작업을 처리하는 클래스"""

    def __init__(self, persist_directory: str, embedding_model: str):
        """
        ChromaDB 클라이언트 및 임베딩 함수 초기화
        """
        # 로거 초기화
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # 로깅 레벨 설정 (예: INFO, DEBUG, ERROR)

        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.embedding_function = self._initialize_embedding_function()
        self.client = self._initialize_chroma_client()
        self.vector_collection = self._load_vector_collection("smartstore_faq")

    def _initialize_embedding_function(self):
        """SentenceTransformer 기반 임베딩 함수 초기화"""
        try:
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model,
                device="cpu"
            )
            self.logger.info(f"Embedding function initialized with model: {self.embedding_model}")
            return embedding_function
        except Exception as e:
            raise RuntimeError(f"임베딩 함수 초기화 실패: {e}")

    def _initialize_chroma_client(self):
        """ChromaDB PersistentClient 초기화"""
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            self.logger.info(f"ChromaDB 클라이언트 초기화 완료. 데이터 저장 경로: {self.persist_directory}")
            return client
        except Exception as e:
            raise RuntimeError(f"ChromaDB 클라이언트 초기화 실패: {e}")

    def _load_vector_collection(self, collection_name: str):
        """ChromaDB 벡터 컬렉션 로드"""
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
            self.logger.info(f"Collection '{collection_name}' loaded with {collection.count()} documents")
            return collection
        except ValueError as e:
            self.logger.error(f"Error loading collection '{collection_name}': {e}")
            raise RuntimeError("Vector collection not found. Ensure the embedding preparation module has been executed.")
        
    def get_related_faq(self, query: str, top_k: int = 3):
        """
        사용자 질문과 관련된 상위 3개의 FAQ를 검색

        Args:
            query (str): 사용자 질문.

        Returns:
            List[dict]: 관련된 FAQ 문서 리스트와 유사도 점수.
                        예: [{"question": "질문1", "answer": "답변1", "score": 0.85}, ...]
        """
        try:
            # 쿼리 실행 (상위 3개만 검색)
            results = self.vector_collection.query(
                query_texts=[query],
                n_results=top_k
            )

            # 결과 디버깅 로깅
            self.logger.debug(f"쿼리 결과: {results}")

            # 결과 처리
            if not results or "documents" not in results or len(results["documents"]) == 0:
                self.logger.warning("관련 FAQ를 찾을 수 없습니다.")
                return []

            # 관련 FAQ 리스트 생성
            related_faqs = [
                {
                    "question": document,
                    "answer": metadata.get("answer", "답변 없음"),
                    "score": score,
                }
                for document, metadata, score in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]

            # 유사도 점수를 기준으로 정렬하여 반환
            return sorted(related_faqs, key=lambda x: x["score"])

        except KeyError as e:
            self.logger.error(f"결과 처리 중 KeyError 발생: {e}")
            return []
        except Exception as e:
            self.logger.error(f"FAQ 검색 중 오류 발생: {e}")
            return []
