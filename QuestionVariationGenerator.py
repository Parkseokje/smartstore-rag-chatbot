import random
import nltk
from nltk.corpus import stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class QuestionVariationGenerator:
    """질문을 변형하는 클래스"""
    
    def __init__(self, language='korean'):
        """
        질문 변형 생성기 초기화
        
        Args:
            language (str): 언어 설정 ('korean' 또는 'english')
        """
        self.language = language
        # 한국어 불용어 목록
        self.korean_stopwords = [
            '이', '그', '저', '것', '이것', '그것', '저것', '는', '은', '이', '가', '을', '를', 
            '에', '의', '로', '으로', '와', '과', '이나', '나', '또는', '혹은', '어떻게', '어떤', 
            '무엇', '어디', '언제', '왜', '하다', '있다', '되다', '같다'
        ]
        
        if language == 'english':
            try:
                self.stopwords = set(stopwords.words('english'))
            except:
                nltk.download('stopwords')
                self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = set(self.korean_stopwords)
    
    def generate_variations(self, question, num_variations=3):
        """
        주어진 질문에 대한 변형을 생성
        
        Args:
            question (str): 원본 질문
            num_variations (int): 생성할 변형 수
            
        Returns:
            list: 변형된 질문 목록
        """
        variations = []
        
        # 변형 방법 목록
        variation_methods = [
            self.word_removal,
            self.word_order_change,
            self.add_typos,
            self.add_question_prefix,
            self.change_punctuation
        ]
        
        # 선택된 방법으로 변형 생성
        selected_methods = random.sample(
            variation_methods, 
            min(num_variations, len(variation_methods))
        )
        
        for method in selected_methods:
            variation = method(question)
            if variation != question and variation not in variations:
                variations.append(variation)
        
        # 필요한 경우 조합 방법 사용
        while len(variations) < num_variations:
            # 두 가지 방법 조합
            method1, method2 = random.sample(variation_methods, 2)
            variation = method2(method1(question))
            
            if variation != question and variation not in variations:
                variations.append(variation)
            
            # 안전 장치 - 너무 많은 시도를 방지
            if len(variations) < num_variations and random.random() < 0.5:
                break
        
        return variations[:num_variations]
    
    def word_removal(self, text):
        """불용어 제거 변형"""
        words = text.split()
        if len(words) <= 3:  # 너무 짧은 문장은 단어를 제거하지 않음
            return text
            
        # 불용어 또는 짧은 단어 찾기
        candidates = []
        for i, word in enumerate(words):
            if word.lower() in self.stopwords or len(word) <= 1:
                candidates.append(i)
        
        if not candidates:
            # 불용어가 없으면 랜덤하게 단어 하나 선택
            candidates = list(range(len(words)))
        
        # 단어 하나 제거
        remove_idx = random.choice(candidates)
        words.pop(remove_idx)
        
        return ' '.join(words)
    
    def word_order_change(self, text):
        """단어 순서 변경 변형"""
        words = text.split()
        if len(words) <= 3:
            return text
            
        # 두 단어 위치 교환
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def add_typos(self, text):
        """오타 추가 변형"""
        words = text.split()
        if not words:
            return text
            
        # 변경할 단어 선택
        word_idx = random.randrange(len(words))
        word = words[word_idx]
        
        if len(word) <= 1:
            return text
        
        # 오타 유형 선택 (글자 변경, 생략, 추가, 위치 변경)
        typo_type = random.randint(0, 3)
        
        if typo_type == 0 and len(word) > 0:  # 글자 변경
            char_idx = random.randrange(len(word))
            chars = list(word)
            # 한글인 경우 비슷한 글자로 변경
            if self.language == 'korean' and '가' <= chars[char_idx] <= '힣':
                similar_chars = {
                    '가': '카', '나': '라', '다': '타', '라': '나', '마': '바', 
                    '바': '파', '사': '싸', '아': '어', '자': '차', '카': '가', 
                    '타': '다', '파': '바', '하': '카'
                }
                if chars[char_idx] in similar_chars:
                    chars[char_idx] = similar_chars[chars[char_idx]]
                else:
                    # 랜덤 한글 생성
                    chars[char_idx] = chr(random.randint(ord('가'), ord('힣')))
            else:
                # 영문 또는 기타 문자
                replacement = chr(random.randint(97, 122))  # a-z
                chars[char_idx] = replacement
            
            words[word_idx] = ''.join(chars)
            
        elif typo_type == 1 and len(word) > 1:  # 글자 생략
            char_idx = random.randrange(len(word))
            words[word_idx] = word[:char_idx] + word[char_idx+1:]
            
        elif typo_type == 2:  # 글자 추가
            char_idx = random.randrange(len(word) + 1)
            # 한글 또는 영문 선택
            if self.language == 'korean' and random.random() < 0.7:
                new_char = chr(random.randint(ord('가'), ord('힣')))
            else:
                new_char = chr(random.randint(97, 122))  # a-z
            words[word_idx] = word[:char_idx] + new_char + word[char_idx:]
            
        elif typo_type == 3 and len(word) > 1:  # 글자 위치 변경
            char_idx = random.randrange(len(word) - 1)
            chars = list(word)
            chars[char_idx], chars[char_idx+1] = chars[char_idx+1], chars[char_idx]
            words[word_idx] = ''.join(chars)
        
        return ' '.join(words)
    
    def add_question_prefix(self, text):
        """질문 접두사 추가 변형"""
        # 한국어/영어 접두사
        prefixes = {
            'korean': [
                '혹시 ', '실례지만 ', '저기요, ', '궁금한데 ', '알려주세요, ', 
                '여쭤보고 싶은데 ', '질문이 있어요. ', '안녕하세요, '
            ],
            'english': [
                'I wonder ', 'Could you tell me ', 'I want to know ', 
                'Please explain ', 'Can you help with ', 'How do I '
            ]
        }
        
        # 이미 접두사가 있는지 확인
        if self.language == 'korean':
            has_prefix = any(text.startswith(p.strip()) for p in prefixes['korean'])
        else:
            has_prefix = any(text.startswith(p.strip()) for p in prefixes['english'])
        
        if has_prefix:
            return text
            
        # 접두사 추가
        prefix = random.choice(prefixes[self.language])
        
        # 문장의 첫 글자를 소문자로 변경 (영어의 경우)
        if self.language == 'english' and text and text[0].isupper():
            text = text[0].lower() + text[1:]
            
        return prefix + text
    
    def change_punctuation(self, text):
        """구두점 변경 변형"""
        # 마지막 구두점 확인
        if text.endswith('.'):
            return text[:-1] + '?'
        elif text.endswith('?'):
            return text[:-1] + '.'
        elif not (text.endswith('.') or text.endswith('?') or text.endswith('!')):
            return text + random.choice(['?', '.', ''])
        return text    