# RAG-GPT: 문서 기반 AI 어시스턴트 🤖

RAG(Retrieval Augmented Generation) 기반의 문서 질의응답 시스템입니다.  
여러 개의 PDF 문서를 한 번에 로드하고, 그 내용을 기반으로 질문에 답변할 수 있습니다.  
CLI(터미널) 모드와 웹(Web) 인터페이스를 모두 지원합니다.

---

## ✨ 주요 기능

- **다중 PDF 지원**
  - 여러 PDF를 동시에 로드하여 통합 검색
  - 문서별 출처 정보 표시
- **두 가지 사용 방식**
  - 💻 CLI REPL 모드 (터미널에서 대화형)
  - 🌐 Gradio 웹 UI (브라우저에서 사용)
- **대화 세션 관리**
  - 대화 내용 저장 및 로드
  - 세션별 히스토리 관리
- **구성 설정 가능**
  - 모델명, temperature, chunk 크기 등 설정 파일로 제어

---

## 📦 설치 방법

### 1. 저장소 클론

git clone https://github.com/Junsoo-Song/rag-gpt.git

cd rag-gpt

### 2. 가상환경 생성 (권장)

python -m venv venv
source venv/bin/activate      # Linux / macOS

### 3. 패키지 설치

pip install -r requirements.txt

### 4. Groq API 키 설정
### 4.1. .env 파일 생성:
cp .env.example .env

### 4.2. .env 파일 열고 다음과 같이 수정:
GROQ_API_KEY=your_groq_api_key_here
또는, 터미널에서 직접 환경 변수로 설정할 수도 있습니다:

export GROQ_API_KEY="your_groq_api_key_here"

## 💻 사용 방법 (CLI 모드)
### 1. 기본 REPL 모드 실행

python -m rag_gpt --repl

실행하면 다음과 같은 안내가 보입니다:

🤖 RAG-GPT REPL 모드
종료: 'exit', 'quit', 또는 Ctrl+D
명령어: !help 로 확인
>

### 2. REPL 명령어 정리
REPL 안에서 사용할 수 있는 명령어:

#### !help
사용 가능한 명령어와 예시를 보여줍니다.

#### !pdf "파일명.pdf"
단일 PDF 파일을 로드합니다.

예:

> !pdf "NGS_Genetic_Testing_Basics.pdf"
!pdfs "파일1.pdf" "파일2.pdf" ...
여러 개의 PDF 파일을 한 번에 로드합니다.
파일명에 공백이 있을 경우 반드시 따옴표(" ")로 감싸야 합니다.

예:

> !pdfs "Distribution of Taste Receptors in Submandibular and von Ebner Salivary Glands.pdf" "NGS_Genetic_Testing_Basics.pdf"

#### !list
현재 메모리에 로드된 PDF 파일 목록을 보여줍니다.

#### !cleardocs
로드된 모든 문서를 초기화합니다 (벡터스토어 리셋).

#### !clear
대화 히스토리(질문/답변)를 초기화합니다.

#### !model 모델이름
사용할 LLM 모델을 변경합니다.

> !model llama-3.3-70b-versatile
exit, quit, 종료
REPL 모드 종료.

## 🌐 사용 방법 (Web 모드)
### 1. 웹 인터페이스 실행

python -m rag_gpt --web
성공적으로 실행되면:

🌐 웹 인터페이스 시작 (포트: 7860)
웹 인터페이스를 시작합니다...
브라우저에서 http://localhost:7860 으로 접속하세요.
* Running on local URL:  http://0.0.0.0:7860
브라우저에서 다음 주소로 접속합니다:

http://localhost:7860
또는
http://127.0.0.1:7860

### 2. 웹 UI 사용 흐름
좌측 상단의 "📄 PDF 업로드 (여러 개 선택 가능)" 영역에서
로컬 PDF 파일들을 하나 이상 선택합니다.

Ctrl + 클릭 또는 Shift + 클릭으로 여러 개 선택 가능
"📥 문서 로드" 버튼 클릭

각 파일당 청크 개수와 총 청크 수가 상태창에 출력됩니다.
아래의 "📚 로드된 문서" 영역에서 현재 로드된 파일 목록을 확인할 수 있습니다.
우측의 채팅창에 질문을 입력하고 Enter 또는 🚀 전송 버튼 클릭

로드된 모든 PDF를 기반으로 관련 내용을 검색하여 답변합니다.
답변 내부에 [출처: 파일명.pdf] 형식으로 어떤 문서에서 인용했는지 표시됩니다.
아래 기능들도 사용할 수 있습니다:

🗑️ 대화 초기화: 현재 채팅 히스토리만 지우고 문서는 유지
🗑️ 문서 초기화: 로드된 PDF/벡터스토어를 초기화
모델 선택 드롭다운: 다른 Groq 모델로 변경
Temperature 슬라이더: 답변의 창의성 정도 조절

### ⚙️설정 파일
전역 설정 파일 위치
처음 실행 시 다음 경로에 기본 설정 파일이 생성됩니다:


~/.rag_gpt/config.json
예시 내용:

{
  "api_key": "",
  "model": "llama-3.3-70b-versatile",
  "temperature": 0.3,
  "chunk_size": 500,
  "chunk_overlap": 50,
  "top_k": 3
}
여기서:

model: 사용할 Groq LLM 모델 이름
temperature: 0.0(매우 보수적) ~ 1.0(매우 창의적)
chunk_size: 문서 청크 크기 (문자 수 기준)
chunk_overlap: 청크 간 겹치는 문자 수
top_k: 검색 시 가져올 상위 청크 개수

### 🛠️ 기술 스택
LangChain: LLM 오케스트레이션 및 체인 구성
Groq: ChatGroq를 통한 LLM 호출
FAISS: 문서 벡터 검색 엔진
HuggingFace Embeddings: intfloat/multilingual-e5-small
Gradio: 웹 UI
Rich: 터미널 출력 포맷팅


