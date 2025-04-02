import openai
from openai import AsyncOpenAI
from typing import List, Dict, AsyncGenerator
import json

class LLMService:
    """OpenAI LLM 서비스를 처리하는 클래스"""

    def __init__(self, api_key: str, llm_model: str):
        """
        OpenAI 클라이언트 초기화
        """
        openai.api_key = api_key
        self.sync_client = openai
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.llm_model = llm_model

    def create_answer_prompt(self, user_message: str, faq_context: str) -> str:
        """사용자 질문과 FAQ 컨텍스트를 기반으로 프롬프트 생성"""
        return f"""
        다음은 네이버 스마트스토어 FAQ에서 검색된 정보입니다:
        {faq_context}
        
        위 정보를 바탕으로 사용자의 질문에 답변해주세요: {user_message}
        
        답변 규칙:
        1. FAQ 내용에 있는 정보만 사용하여 답변합니다.
        2. 정보가 없는 경우, 없다고 정직하게 답변합니다.
        3. 친절하고 도움이 되는 톤으로 답변합니다.
        4. 답변은 간결하고 명확하게 작성합니다.
        """        

    def is_smartstore_related(self, query: str) -> bool:
        """질문이 스마트스토어와 관련 있는지 확인"""
        system_prompt = """
        당신은 네이버 스마트스토어 FAQ를 담당하는 챗봇입니다.
        사용자의 질문이 스마트스토어와 관련이 있는지 판단해주세요.
        관련이 있으면 True, 없으면 False로만 대답해주세요.
        """
        try:
            response = self.sync_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"다음 질문이 네이버 스마트스토어와 관련 있나요? 질문: {query}"},
                ],
                temperature=0.1,
                max_tokens=10,
            )
            result = response.choices[0].message.content.strip().lower()
            return "true" in result
        except Exception as e:
            print(f"Error checking if query is related to smartstore: {e}")
            return True

    def generate_follow_up_questions(self, context: str, conversation: List[Dict[str, str]]) -> List[str]:
        """대화 내용을 기반으로 후속 질문 생성"""
        system_prompt = """
        당신은 네이버 스마트스토어 FAQ 챗봇입니다.
        현재 대화 맥락에서 사용자가 다음으로 물어볼 만한 2가지 관련 질문을 생성해주세요.
        질문 형태로 작성해주세요. 각 질문을 별도의 줄에 작성하고, 머리글이나 번호 없이 질문만 작성해주세요.
        """
        try:
            response = self.sync_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"다음은 현재까지의 대화 내용입니다:\n{json.dumps(conversation, ensure_ascii=False)}\n\n다음은 검색된 FAQ 정보입니다:\n{context}\n\n사용자가 다음으로 물어볼 만한 2가지 질문을 생성해주세요.",
                    },
                ],
                temperature=0.7,
                max_tokens=150,
            )
            follow_up_text = response.choices[0].message.content.strip()
            follow_up_questions = [q.strip() for q in follow_up_text.split("\n") if q.strip()]
            return follow_up_questions[:2]  # 최대 2개 질문 반환
        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
            return [
                "다른 스마트스토어 관련 문의가 있으신가요?",
                "더 알고 싶은 정보가 있으신가요?",
            ]

    def generate_answer(self, conversation_history: List[Dict[str, str]], prompt: str) -> str:
        """OpenAI API를 사용하여 동기적으로 답변 생성"""
        try:
            response = self.sync_client.chat.completions.create(
                model=self.llm_model,
                messages=conversation_history + [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다. 다시 질문해주시겠어요?"

    async def generate_streaming_answer(self, conversation_history: List[Dict[str, str]], prompt: str) -> AsyncGenerator[str, None]:
        """OpenAI API를 사용하여 스트리밍 응답 생성"""
        try:
            response_stream = await self.async_client.chat.completions.create(
                model=self.llm_model,
                messages=conversation_history + [{"role": "user", "content": prompt}],
                temperature=0.1,
                stream=True,
            )
            async for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error in streaming answer generation: {e}")
            yield "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다. 다시 질문해주시겠어요?"
