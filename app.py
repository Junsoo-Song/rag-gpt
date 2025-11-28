"""
메인 애플리케이션 로직
"""
import os
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown

from .handlers.chat_handler import ChatHandler
from .handlers.rag_handler import RAGHandler
from .cache import VectorCache
from .config import Config

console = Console()

class RagGPT:
    """메인 애플리케이션 클래스"""
    
    def __init__(self, config: Config, use_cache: bool = True):
        self.config = config
        self.use_cache = use_cache
        
        # API 키를 먼저 설정 (핸들러 초기화 전에!)
        api_key = config.get("api_key")
        if not api_key:
            # .env 파일에서 읽기 시도
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")
            
            if not api_key:
                console.print("[red]GROQ API 키가 설정되지 않았습니다![/red]")
                console.print("다음 중 하나의 방법으로 설정하세요:")
                console.print("1. export GROQ_API_KEY='your-api-key'")
                console.print("2. .env 파일에 GROQ_API_KEY=your-api-key 추가")
                console.print("3. ~/.rag_gpt/config.json 파일에서 api_key 설정")
                raise ValueError("GROQ API key not found")
        
        os.environ["GROQ_API_KEY"] = api_key
        
        # 이제 핸들러 초기화
        self.chat_handler = ChatHandler(config)
        self.rag_handler = RAGHandler(config)
        self.cache = VectorCache() if use_cache else None
    
    def load_pdf(self, pdf_path: Path):
        """PDF 로드 및 벡터화"""
        console.print(f"[cyan]📄 PDF 로딩: {pdf_path}[/cyan]")
        
        # 캐시 확인
        if self.cache and self.cache.exists(pdf_path):
            console.print("[green]✅ 캐시된 벡터스토어 로드[/green]")
            self.rag_handler.load_vectorstore(
                self.cache.load(pdf_path)
            )
        else:
            # 새로 벡터화
            vectorstore = self.rag_handler.process_pdf(pdf_path)
            if self.cache:
                self.cache.save(pdf_path, vectorstore)
    
    def query(self, prompt: str, session_name: Optional[str] = None) -> str:
        """단일 질문 처리"""
        # 세션 로드
        if session_name:
            self.chat_handler.load_session(session_name)
        
        # RAG 처리
        response = self.rag_handler.query(
            prompt, 
            chat_history=self.chat_handler.get_history()
        )
        
        # 히스토리 업데이트
        self.chat_handler.add_message("user", prompt)
        self.chat_handler.add_message("assistant", response)
        
        # 세션 저장
        if session_name:
            self.chat_handler.save_session(session_name)
        
        return response
    
    def start_repl(self, session_name: Optional[str] = None):
        """대화형 REPL 모드"""
        console.print("[bold cyan]🤖 RAG-GPT REPL 모드[/bold cyan]")
        console.print("종료: 'exit', 'quit', 또는 Ctrl+D\n")
        
        # 세션 로드
        if session_name:
            self.chat_handler.load_session(session_name)
            console.print(f"[green]세션 '{session_name}' 로드됨[/green]\n")
        
        while True:
            try:
                # 프롬프트
                prompt = Prompt.ask("[bold yellow]>[/bold yellow]")
                
                if prompt.lower() in ['exit', 'quit', '종료']:
                    break
                
                # 특수 명령어 처리
                if prompt.startswith("!"):
                    self._handle_command(prompt[1:])
                    continue
                
                # 질문 처리
                response = self.query(prompt, session_name)
                
                # 응답 출력
                console.print("\n[bold green]AI:[/bold green]")
                console.print(Markdown(response))
                console.print("\n" + "-"*50 + "\n")
                
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                console.print(f"[red]오류: {e}[/red]")
        
        console.print("\n[cyan]REPL 모드를 종료합니다.[/cyan]")
    
    def _handle_command(self, command: str):
        """특수 명령어 처리"""
        parts = command.split()
        cmd = parts[0] if parts else ""
        
        if cmd == "clear":
            self.chat_handler.clear_history()
            console.print("[yellow]대화 기록 초기화[/yellow]")
        elif cmd == "history":
            self.chat_handler.show_history()
        elif cmd == "pdf" and len(parts) > 1:
            self.load_pdf(Path(parts[1]))
        elif cmd == "model" and len(parts) > 1:
            self.config.set("model", parts[1])
            console.print(f"[green]모델 변경: {parts[1]}[/green]")
        else:
            console.print("[red]알 수 없는 명령어[/red]")
            console.print("사용 가능: !clear, !history, !pdf <file>, !model <name>")
    
    def list_chats(self):
        """대화 목록 표시"""
        self.chat_handler.list_sessions()
    
    def show_chat(self, session_name: Optional[str]):
        """대화 기록 표시"""
        self.chat_handler.show_session(session_name)

