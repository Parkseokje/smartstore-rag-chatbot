import shutil  # 디렉토리 삭제를 위해 추가

# SQLite 모듈 교체 (pysqlite3 사용)
import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3

import os
import pickle
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from dotenv import load_dotenv

# 상수 정의
# 환경 변수 로드 및 설정
load_dotenv()
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
ID_PREFIX = "id"
BATCH_SIZE = 128

# 간단한 로깅 초기화
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VectorPreparation:
    """벡터 데이터베이스 준비를 위한 클래스"""

    def __init__(self, data_path: str):
        # 로거 초기화
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # 로깅 레벨 설정 (예: INFO, DEBUG, ERROR)
        
        self.data_path = data_path
        self.data = None

        # 디렉토리 초기화 (재생성)
        self._reset_directory(CHROMA_PERSIST_DIRECTORY)

        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma_client = PersistentClient(path=os.path.abspath(CHROMA_PERSIST_DIRECTORY))
        self.collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME)

    def _reset_directory(self, directory: str):
        """기존 디렉토리를 삭제하고 새로 생성"""
        if os.path.exists(directory):
            logging.info(f"기존 디렉토리를 삭제합니다: {directory}")
            shutil.rmtree(directory)  # 디렉토리와 내부 파일/폴더 삭제
        os.makedirs(directory, exist_ok=True)  # 빈 디렉토리 생성
        logging.info(f"새로운 디렉토리를 생성했습니다: {directory}")        

    def load_data(self) -> bool:
        """데이터 로드"""
        try:
            with open(self.data_path, "rb") as f:
                raw_data = pickle.load(f)
            self.data = self._validate_and_format_data(raw_data)
            logging.info(f"{len(self.data)}개의 데이터를 성공적으로 로드했습니다.")
            return True
        except FileNotFoundError:
            logging.error(f"파일을 찾을 수 없습니다: {self.data_path}")
            return False
        except Exception as e:
            logging.error(f"데이터 로드 중 오류 발생: {e}")
            return False

    def _validate_and_format_data(self, raw_data):
        """데이터 형식 검증 및 변환"""
        if isinstance(raw_data, dict):
            return [{"question": k, "answer": v} for k, v in raw_data.items()]
        elif isinstance(raw_data, list) and all(
            isinstance(item, dict) and "question" in item and "answer" in item for item in raw_data
        ):
            return raw_data
        else:
            raise ValueError("데이터 형식이 올바르지 않습니다.")

    def embed_in_batches(self) -> bool:
        """데이터를 배치 단위로 임베딩하고 저장"""
        if not self.data:
            logging.error("로드된 데이터가 없습니다.")
            return False

        try:
            existing_count = self.collection.count()
            total_items = len(self.data)
            batch_count = (total_items + BATCH_SIZE - 1) // BATCH_SIZE

            logging.info(f"{total_items}개의 데이터를 {batch_count}개의 배치로 처리합니다.")

            for i in range(0, total_items, BATCH_SIZE):
                batch_data = self.data[i : i + BATCH_SIZE]
                
                # 데이터 임베딩 시 질문과 답변 결합
                questions = [item["question"] for item in batch_data]
                # questions_and_answers = [item["question"] + " " + item["answer"] for item in batch_data]

                embeddings = self.embedding_model.encode(questions, convert_to_numpy=True).tolist()
                ids = [f"{ID_PREFIX}{j + existing_count}" for j in range(i, i + len(batch_data))]
                metadatas = [{"answer": item["answer"], "timestamp": datetime.now().isoformat()} for item in batch_data]

                # 컬렉션에 추가
                self.collection.add(ids=ids, embeddings=embeddings, documents=questions, metadatas=metadatas)

                logging.info(f"{len(batch_data)}개의 데이터를 처리했습니다. (현재 진행률: {i + len(batch_data)}/{total_items})")

            logging.info("모든 데이터를 성공적으로 저장했습니다.")
            return True

        except Exception as e:
            logging.error(f"임베딩 및 저장 중 오류 발생: {e}")
            return False

    def verify_collection(self) -> bool:
        """컬렉션 검증"""
        try:
            count = self.collection.count()
            logging.info(f"컬렉션에 총 {count}개의 항목이 저장되어 있습니다.")

            if count > 0 and self.data:
                sample_query = self.data[0]["question"]
                results = self.collection.query(query_texts=[sample_query], n_results=1)
                if results and results["documents"]:
                    logging.info(f"쿼리 테스트 성공. 샘플 결과: {results['documents'][0]}")
                    return True
                else:
                    logging.warning("쿼리 테스트에서 결과가 반환되지 않았습니다.")
                    return False

            return True

        except Exception as e:
            logging.error(f"컬렉션 검증 중 오류 발생: {e}")
            return False

    def run(self) -> bool:
        """전체 프로세스 실행"""
        start_time = datetime.now()
        logging.info("데이터 임베딩 및 저장 프로세스를 시작합니다.")

        if not self.load_data():
            return False

        if not self.embed_in_batches():
            return False

        verified = self.verify_collection()
        elapsed_time = datetime.now() - start_time
        status_message = "성공" if verified else "실패"
        
        logging.info(f"데이터 임베딩 및 저장 프로세스 {status_message}. (총 소요 시간: {elapsed_time})")
        
        return verified


if __name__ == "__main__":
    processor = VectorPreparation("final_result.pkl")
    if processor.run():
        print("데이터 임베딩 및 저장 완료.")
    else:
        print("데이터 임베딩 및 저장 과정에서 오류가 발생했습니다.")
