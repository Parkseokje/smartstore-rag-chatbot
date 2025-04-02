import csv
import logging
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os
import random

# SQLite 모듈 교체 (pysqlite3 사용)
import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3

from ChromaDBService import ChromaDBService
from QuestionVariationGenerator import QuestionVariationGenerator

# 환경 변수 로드 및 설정
load_dotenv()
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

class ChromaDBValidator:
    """ChromaDB 컬렉션의 검색 성능을 검증하는 클래스"""
    
    def __init__(self, chroma_service, language='korean'):
        """
        ChromaDB 검증기 초기화
        
        Args:
            chroma_service: ChromaDBService 인스턴스
        """
        self.chroma_service = chroma_service
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.variation_generator = QuestionVariationGenerator(language)
        
    def get_all_collection_data(self, collection_name="smartstore_faq"):
        """
        컬렉션의 모든 데이터 가져오기
        
        Returns:
            dict: 컬렉션의 모든 데이터 (documents, metadatas, ids)
        """
        try:
            # 컬렉션 가져오기 (이미 서비스에 로드되어 있으면 그것을 사용)
            if self.chroma_service.vector_collection.name == collection_name:
                collection = self.chroma_service.vector_collection
            else:
                collection = self.chroma_service.client.get_collection(
                    name=collection_name,
                    embedding_function=self.chroma_service.embedding_function
                )
            
            # 모든 데이터 가져오기
            all_data = collection.get()
            self.logger.info(f"컬렉션 '{collection_name}'에서 {len(all_data.get('ids', []))}개 문서를 가져왔습니다.")
            return all_data
            
        except Exception as e:
            self.logger.error(f"컬렉션 데이터 가져오기 실패: {e}")
            raise
    
    def validate_collection(self, collection_name="smartstore_faq", top_k=5,
                            variations_per_question=3, sample_size=None, seed=42):
        """
        컬렉션의 질문-응답 쌍에 대해 검증 수행
        
        Args:
            collection_name (str): 검증할 컬렉션 이름
            top_k (int): 각 쿼리에 대해 검색할 결과 수
            variations_per_question (int): 각 질문에 대해 생성할 변형 수
            sample_size (int, optional): 검증할 질문 샘플 수 (None이면 전체)
            seed (int): 랜덤 시드
            
        Returns:
            list: 검증 결과 리스트
        """
        random.seed(seed)

        # 모든 데이터 가져오기
        all_data = self.get_all_collection_data(collection_name)
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])

        # 샘플링 (필요한 경우)
        if sample_size and sample_size < len(documents):
            # 인덱스 샘플링
            indices = random.sample(range(len(documents)), sample_size)
            documents = [documents[i] for i in indices]
            metadatas = [metadatas[i] for i in indices]
            ids = [ids[i] for i in indices]
            self.logger.info(f"{sample_size}개의 질문을 샘플링하여 검증합니다.")
        
        validation_results = []
        
        # 각 문서(질문)에 대해 검증
        for i, (doc, metadata, doc_id) in enumerate(tqdm(zip(documents, metadatas, ids), desc="원본 질문 검증 중", total=len(documents))):
            # 원본 질문 검증
            original_results = self._validate_single_query(doc, doc_id, metadata, top_k)
            validation_results.append(original_results)
            
            # 질문 변형 생성 및 검증
            variations = self.variation_generator.generate_variations(doc, variations_per_question)
            
            for j, variation in enumerate(variations):
                # 변형된 질문 검증
                variation_results = self._validate_single_query(
                    variation, f"{doc_id}_var{j+1}", metadata, top_k,
                    original_question=doc,
                    is_variation=True
                )
                validation_results.append(variation_results)
        
        return validation_results
    
    def _validate_single_query(self, query, query_id, metadata, top_k, original_question=None, is_variation=False):
        """
        단일 쿼리에 대한 검증 수행
        
        Args:
            query (str): 검증할 쿼리 (질문)
            query_id (str): 쿼리 ID
            metadata (dict): 질문 메타데이터
            top_k (int): 검색할 결과 수
            original_question (str, optional): 원본 질문 (변형인 경우)
            is_variation (bool): 변형 여부
            
        Returns:
            dict: 검증 결과
        """
        # 원본 답변 가져오기
        original_answer = metadata.get("answer", "")
        
        # 검색 수행
        results = self.chroma_service.get_related_faq(query, top_k)
        
        # 원본 문서가 결과에 포함되는지 확인
        found = False
        rank = -1
        similarity = 0.0
        
        for idx, result in enumerate(results):
            if not is_variation and result["question"] == query:
                # 원본 질문 검증
                found = True
                rank = idx + 1
                similarity = result["score"]
                break
            elif is_variation and result["question"] == original_question:
                # 변형 질문 검증 (원본 질문이 결과에 있는지)
                found = True
                rank = idx + 1
                similarity = result["score"]
                break
        
        # 검증 결과 저장
        validation_result = {
            "id": query_id,
            "question": query,
            "original_question": original_question if is_variation else query,
            "is_variation": is_variation,
            "original_answer": original_answer,
            "found_in_results": found,
            "rank": rank if found else -1,
            "similarity": similarity,
            "top_result_question": results[0]["question"] if results else "",
            "top_result_answer": results[0]["answer"] if results else "",
            "top_result_similarity": results[0]["score"] if results else 0.0
        }
        
        return validation_result
    
    def calculate_metrics(self, validation_results):
        """
        검증 결과에서 메트릭 계산
        
        Args:
            validation_results (list): 검증 결과 리스트
            
        Returns:
            dict: 계산된 메트릭
        """
        # 원본 질문과 변형 질문 분리
        original_results = [r for r in validation_results if not r["is_variation"]]
        variation_results = [r for r in validation_results if r["is_variation"]]
        
        # 원본 질문 메트릭
        total_original = len(original_results)
        original_found = sum(1 for r in original_results if r["found_in_results"])
        
        # 변형 질문 메트릭
        total_variations = len(variation_results)
        variations_found = sum(1 for r in variation_results if r["found_in_results"])
        
        # 통합 메트릭
        metrics = {
            # 원본 질문 메트릭
            "total_original_questions": total_original,
            "original_questions_found": original_found,
            "original_retrieval_accuracy": original_found / total_original if total_original > 0 else 0,
            "original_avg_similarity": sum(r["similarity"] for r in original_results) / total_original if total_original > 0 else 0,
            "original_top1_accuracy": sum(1 for r in original_results if r["rank"] == 1) / total_original if total_original > 0 else 0,
            "original_mrr": sum(1/r["rank"] if r["rank"] > 0 else 0 for r in original_results) / total_original if total_original > 0 else 0,
            
            # 변형 질문 메트릭
            "total_variation_questions": total_variations,
            "variation_questions_found": variations_found,
            "variation_retrieval_accuracy": variations_found / total_variations if total_variations > 0 else 0,
            "variation_avg_similarity": sum(r["similarity"] for r in variation_results) / total_variations if total_variations > 0 else 0,
            "variation_top1_accuracy": sum(1 for r in variation_results if r["rank"] == 1) / total_variations if total_variations > 0 else 0,
            "variation_mrr": sum(1/r["rank"] if r["rank"] > 0 else 0 for r in variation_results) / total_variations if total_variations > 0 else 0,
            
            # 전체 메트릭
            "total_questions": len(validation_results),
            "total_questions_found": original_found + variations_found,
            "overall_retrieval_accuracy": (original_found + variations_found) / len(validation_results) if validation_results else 0,
            "overall_avg_similarity": sum(r["similarity"] for r in validation_results) / len(validation_results) if validation_results else 0,
            "overall_top1_accuracy": sum(1 for r in validation_results if r["rank"] == 1) / len(validation_results) if validation_results else 0,
            "overall_mrr": sum(1/r["rank"] if r["rank"] > 0 else 0 for r in validation_results) / len(validation_results) if validation_results else 0
        }
        
        # 변형 유형별 성능 (나중에 구현 가능)
        
        return metrics
    
    def save_results_to_csv(self, validation_results, metrics, output_file="chromadb_validation_results.csv"):
        """
        검증 결과를 간결한 CSV 파일로 저장
        
        Args:
            validation_results (list): 검증 결과 리스트
            metrics (dict): 계산된 메트릭
            output_file (str): 출력 파일 이름
        """
        try:
            # 결과를 데이터프레임으로 변환하고 필요한 컬럼만 선택
            df = pd.DataFrame(validation_results)
            
            # 답변 내용 일부만 표시하도록 처리
            df['original_answer'] = df['original_answer'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
            df['top_result_answer'] = df['top_result_answer'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
            
            # 중요 컬럼만 선택
            columns = ['id', 'question', 'is_variation', 'found_in_results', 'rank', 
                    'similarity', 'top_result_question', 'top_result_similarity']
            
            # CSV 파일로 저장
            with open(output_file, mode="w", newline="", encoding="utf-8") as file:
                # 메트릭 정보 - 간결하게 표시
                file.write("# 메트릭 요약\n")
                
                # 주요 메트릭만 표시
                key_metrics = {
                    '원본_검색정확도': metrics['original_retrieval_accuracy'],
                    '변형_검색정확도': metrics['variation_retrieval_accuracy'],
                    '전체_검색정확도': metrics['overall_retrieval_accuracy'],
                    '원본_Top1정확도': metrics['original_top1_accuracy'],
                    '변형_Top1정확도': metrics['variation_top1_accuracy'],
                    '원본_MRR': metrics['original_mrr'],
                    '변형_MRR': metrics['variation_mrr']
                }
                
                metrics_writer = csv.writer(file)
                metrics_writer.writerow(key_metrics.keys())
                metrics_writer.writerow([f"{v:.4f}" for v in key_metrics.values()])
                
                # 빈 줄 추가
                file.write("\n# 검증 결과\n")
                
                # 선택된 컬럼만으로 메인 결과 데이터 저장
                df[columns].to_csv(file, index=False)
            
            self.logger.info(f"간결한 검증 결과가 '{output_file}'에 저장되었습니다.")
                
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류 발생: {e}")
            raise

# 사용 예시
def run_validation():
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    language = "korean"  # 'korean' 또는 'english'
    variations_per_question = 3  # 각 질문당 변형 수
    sample_size = 10  # 검증할 질문 샘플 수 (None이면 전체)
    
    try:
        # ChromaDBService 초기화
        chroma_service = ChromaDBService(CHROMA_PERSIST_DIRECTORY, EMBEDDING_MODEL)
        
        # 검증기 초기화
        validator = ChromaDBValidator(chroma_service)
        
        # 검증 수행
        validation_results = validator.validate_collection(
            collection_name=COLLECTION_NAME,
            top_k=5,
            variations_per_question=variations_per_question,
            sample_size=sample_size
        )
        
        # 메트릭 계산
        metrics = validator.calculate_metrics(validation_results)
        
        # 결과 저장
        validator.save_results_to_csv(validation_results, metrics)
        
        # 결과 요약 출력
        print(f"\n검증 결과 요약:")
        print(f"원본 질문 검색 정확도: {metrics['original_retrieval_accuracy']:.2%}")
        print(f"변형 질문 검색 정확도: {metrics['variation_retrieval_accuracy']:.2%}")
        print(f"전체 검색 정확도: {metrics['overall_retrieval_accuracy']:.2%}")
        print(f"원본 Top-1 정확도: {metrics['original_top1_accuracy']:.2%}")
        print(f"변형 Top-1 정확도: {metrics['variation_top1_accuracy']:.2%}")
        print(f"원본 MRR: {metrics['original_mrr']:.4f}")
        print(f"변형 MRR: {metrics['variation_mrr']:.4f}")
        
    except Exception as e:
        logging.error(f"검증 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    run_validation()