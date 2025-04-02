import uuid
from typing import List, Dict, Any, Optional
import os
import json
import asyncio
import logging

from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from contextlib import asynccontextmanager

# SQLite3 패치
import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 환경 변수 로드 및 설정
load_dotenv()
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

# 외부 서비스 임포트
from LLMService import LLMService
from ChromaDBService import ChromaDBService

## 유틸리티 함수

def get_session_id(session_id: Optional[str] = None) -> str:
    """세션 ID를 가져오거나 새로 생성"""
    if session_id:
        logger.info(f"기존 세션 ID 사용: {session_id}")
        return session_id
    
    new_session_id = str(uuid.uuid4())
    logger.info(f"새로운 세션 ID 생성: {new_session_id}")
    return new_session_id


def initialize_services(app: FastAPI):
    """서비스 초기화"""
    if hasattr(app.state, "llm_service") and hasattr(app.state, "chroma_service"):
        logger.info("서비스가 이미 초기화되었습니다.")
        return  # 중복 초기화를 방지
    
    logger.info("서비스 초기화 시작...")
    app.state.llm_service = LLMService(api_key=OPENAI_API_KEY, llm_model=LLM_MODEL)
    app.state.chroma_service = ChromaDBService(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_model=EMBEDDING_MODEL,
    )
    logger.info("서비스 초기화 완료.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱의 시작 및 종료 이벤트 핸들러"""
    logger.info("애플리케이션 시작 중...")
    
    if not hasattr(app.state, "initialized"):
        initialize_services(app)
        app.state.initialized = True  # 초기화 상태 저장
    
    yield  # 애플리케이션 실행 중간에 실행될 코드
    
    logger.info("애플리케이션 종료 중...")

# FastAPI 앱 초기화 및 설정
app = FastAPI(title="Smartstore FAQ Chatbot API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 세션 스토리지 (실제 구현에서는 데이터베이스 사용 권장)
conversation_history: Dict[str, List[Dict[str, str]]] = {}


## 모델 정의

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    follow_up_questions: List[str]
    done: bool


def handle_unrelated_message(session_id: str) -> Dict[str, Any]:
    """스마트스토어와 관련 없는 메시지 처리"""
    
    logger.info(f"세션 {session_id}: 스마트스토어와 관련 없는 메시지 처리 중...")
    
    answer = "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."
    follow_up_questions = [
        "스마트스토어 회원가입에 대해 알고 싶으신가요?",
        "스마트스토어에서 판매할 수 있는 상품에 대해 알고 싶으신가요?",
    ]
    
    conversation_history[session_id].append({"role": "assistant", "content": answer})
    
    logger.info(f"세션 {session_id}: 스마트스토어와 관련 없는 메시지 응답 생성 완료.")
    
    return {
        "session_id": session_id,
        "answer": answer,
        "follow_up_questions": follow_up_questions,
        "done": True
    }

# 헬퍼 함수들
def initialize_session_if_needed(session_id: str):
    """세션이 존재하지 않으면 초기화"""
    if session_id not in conversation_history:
        conversation_history[session_id] = [
            {"role": "system", "content": "당신은 네이버 스마트스토어 FAQ를 담당하는 챗봇입니다. FAQ 내용을 기반으로 정확하고 도움이 되는 답변을 제공해주세요."}
        ]
        logger.info(f"세션 {session_id}: 초기 대화 기록 생성 완료.")

def add_message_to_history(session_id: str, role: str, content: str):
    """대화 기록에 메시지 추가"""
    conversation_history[session_id].append({"role": role, "content": content})
    logger.info(f"세션 {session_id}: {role} 메시지 추가 완료.")    

def get_relevant_faqs(chroma_service: ChromaDBService, session_id: str, message: str):
    """관련 FAQ를 검색하고 로깅"""
    try:
        relevant_faqs = chroma_service.get_related_faq(message)
        logger.info(f"세션 {session_id}: 관련 FAQ 검색 결과: {json.dumps(relevant_faqs, ensure_ascii=False, indent=2)}")
        return relevant_faqs
    except Exception as e:
        logger.error(f"세션 {session_id}: 관련 FAQ 검색 중 오류 발생: {str(e)}")
        return []        
    
def create_faq_context(relevant_faqs: list) -> str:
    """관련 FAQ 목록으로부터 컨텍스트 문자열 생성"""
    if not relevant_faqs:
        return "관련된 FAQ를 찾을 수 없습니다."
    
    return "\n\n".join(
        [f"질문: {faq['question']}\n답변: {faq['answer']}\n유사도 점수: {faq['score']}" for faq in relevant_faqs]
    )    
    
async def stream_unrelated_response(session_id: str):
    """스마트스토어와 관련 없는 질문에 대한 응답 스트리밍"""
    response_data = handle_unrelated_message(session_id)
    
    # 먼저 answer와 follow_up_questions를 스트리밍
    yield f"data: {json.dumps({'session_id': response_data['session_id'], 'answer': response_data['answer'], 'done': False}, ensure_ascii=False)}\n\n"

    # 마지막으로 done: True 전송
    yield f"data: {json.dumps({'session_id': response_data['session_id'], 'follow_up_questions': response_data['follow_up_questions'], 'done': True}, ensure_ascii=False)}\n\n"    

async def stream_related_response(session_id: str, user_message: str, relevant_faqs: list, llm_service: LLMService):
    """관련 FAQ를 기반으로 응답 스트리밍"""
    full_answer = ""
    
    # FAQ 컨텍스트 구성
    faq_context = create_faq_context(relevant_faqs)
    
    # 프롬프트 생성 및 응답 스트리밍
    prompt = llm_service.create_answer_prompt(user_message, faq_context)
    logger.info(f"prompt: {prompt}")
    
    async for content in llm_service.generate_streaming_answer(conversation_history[session_id], prompt):
        full_answer += content
        yield f"data: {json.dumps({'session_id': session_id, 'answer': content, 'done': False}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.01)
    
    # 응답 저장
    add_message_to_history(session_id, "assistant", full_answer)
    
    # 후속 질문 생성
    follow_up_questions = llm_service.generate_follow_up_questions(faq_context, conversation_history[session_id])
    
    # 최종 응답
    final_response = {
        "session_id": session_id,
        "answer": "",
        "done": True,
        "follow_up_questions": follow_up_questions,
    }
    
    yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n"
    logger.info(f"세션 {session_id}: 스트리밍 응답 및 후속 질문 완료.")

## 엔드포인트 정의

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """HTML 템플릿 렌더링"""
    
    logger.info("HTML 템플릿 요청 처리 중...")

    # 초기 챗봇 메시지 추가
    initial_message = "안녕하세요! 스마트스토어 FAQ 챗봇입니다. 무엇을 도와드릴까요?"

    return templates.TemplateResponse(
        "chat.html", {"request": request, "initial_message": initial_message}
    )

@app.get("/stream-chat")
async def stream_chat(message: str, session_id: Optional[str] = None):
    """HTTP 스트리밍 채팅 엔드포인트"""
    
    chroma_service: ChromaDBService = app.state.chroma_service
    llm_service: LLMService = app.state.llm_service
    
    # 세션 관리
    session_id = get_session_id(session_id)
    initialize_session_if_needed(session_id)
    
    # 사용자 메시지 처리
    user_message = message.strip()
    add_message_to_history(session_id, "user", user_message)

    # 스마트스토어 관련 질문인지 확인
    if not llm_service.is_smartstore_related(user_message):
        logger.info(f"세션 {session_id}: 스마트스토어와 관련 없는 메시지 스트리밍 시작.")
        return StreamingResponse(
            stream_unrelated_response(session_id),
            media_type="text/event-stream"
        )

    # 관련 FAQ 가져오기
    relevant_faqs = get_relevant_faqs(chroma_service, session_id, user_message)
    
    # 응답 스트리밍
    logger.info(f"세션 {session_id}: FAQ 컨텍스트 및 프롬프트 생성 완료.")
    return StreamingResponse(
        stream_related_response(session_id, user_message, relevant_faqs, llm_service),
        media_type="text/event-stream"
    )
