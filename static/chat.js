// chat.js
let sessionId = null; // To maintain conversation state

document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
document.addEventListener('DOMContentLoaded', function() {
    const chatBox = document.getElementById('chat-box');
    
    // 초기 인사말 추가
    const initialMessage = document.createElement('div');
    initialMessage.classList.add('bot-message'); // 스타일을 위한 클래스 추가
    initialMessage.textContent = "안녕하세요! 스마트스토어 FAQ 챗봇입니다. 무엇을 도와드릴까요?";
    
    chatBox.appendChild(initialMessage);
});

async function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (!userInput.trim()) return;
    
    appendMessage('사용자', userInput);
    document.getElementById('user-input').value = '';
    
    try {
        // 임시 "대기 중" 메시지 표시
        const waitingId = appendMessage('챗봇', '입력 중...');
        
        // Construct proper URL with query parameters
        const url = `/stream-chat?message=${encodeURIComponent(userInput)}${sessionId ? `&session_id=${sessionId}` : ''}`;
        
        const response = await fetch(url, {
            headers: {
                'Accept': 'text/event-stream',
            },
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        // Remove waiting message
        removeMessage(waitingId);
        
        // Process the event stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        
        let fullAnswer = '';
        let botMessageId = null;
        let buffer = '';
        
        while (true) {
            const { value, done } = await reader.read();            
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // Process complete server-sent events
            const events = buffer.split('\n\n');
            buffer = events.pop() || ''; // Keep the last incomplete chunk
            
            for (const event of events) {
                if (event.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(event.substring(6));
                        
                        // Save session ID for future requests
                        if (data.session_id) {
                            sessionId = data.session_id;
                        }
                        
                        if (data.done) {
                            // Show follow-up questions if available
                            if (data.follow_up_questions && data.follow_up_questions.length > 0) {
                                appendFollowUpQuestions(data.follow_up_questions);
                            }
                        } else if (data.answer) {
                            // Update streaming response
                            if (!botMessageId) {
                                botMessageId = appendMessage('챗봇', data.answer);
                                fullAnswer = data.answer;
                            } else {
                                fullAnswer += data.answer;
                                updateMessage(botMessageId, fullAnswer);
                            }
                        }
                    } catch (e) {
                        console.error('Error parsing SSE data:', e, event.substring(6));
                    }
                }
            }
        }
    } catch (error) {
        console.error('Fetch error:', error);
        appendMessage('챗봇', '오류가 발생했습니다. 다시 시도해주세요.');
    }
}

// 메시지를 추가하고 해당 메시지의 ID를 반환
function appendMessage(sender, message) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.classList.add(sender === '사용자' ? 'user-message' : 'bot-message');
    messageElement.textContent = `${sender}: ${message}`;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;

    // 고유한 ID 생성: sender 정보 포함
    const id = `${sender === '사용자' ? 'user' : 'bot'}-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
    messageElement.id = id;

    return id;
}


// ID로 메시지 제거
function removeMessage(id) {
    const element = document.getElementById(id);
    if (
        element &&
        element.textContent.trim() === "챗봇: 입력 중..." &&
        element.classList.contains("bot-message") // 클래스 이름 확인
    ) {
        element.remove();
    }
}

// ID로 메시지 내용 업데이트
// ID로 메시지 내용 업데이트
function updateMessage(id, message) {
    const element = document.getElementById(id);
    if (element) {
        const sender = element.classList.contains('user-message') ? '사용자' : '챗봇';
        const formattedMessage = marked.parse(message); // 마크다운을 HTML로 변환
        element.innerHTML = `${sender}: ${formattedMessage}`; // innerHTML 사용
        const chatBox = document.getElementById('chat-box');
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

// 후속 질문 버튼 추가
function appendFollowUpQuestions(questions) {
    const chatBox = document.getElementById('chat-box');
    const followUpContainer = document.createElement('div');
    followUpContainer.classList.add('follow-up-container');
    
    const label = document.createElement('div');
    label.textContent = '💡 추천 질문:';
    label.classList.add('follow-up-label');
    followUpContainer.appendChild(label);
    
    questions.forEach(question => {
        const button = document.createElement('button');
        button.textContent = question;
        button.classList.add('follow-up-button');
        button.addEventListener('click', () => {
            document.getElementById('user-input').value = question;
            sendMessage();
        });
        followUpContainer.appendChild(button);
    });
    
    chatBox.appendChild(followUpContainer);
    chatBox.scrollTop = chatBox.scrollHeight;
}