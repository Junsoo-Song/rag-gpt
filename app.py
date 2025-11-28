"""
메인 애플리케이션 로직 - 다중 PDF 지원
"""
import os
import shlex
from pathlib import Path
from typing import Optional, List, Union
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
        
        # API 키 설정
        api_key = config.get("api_key")
        if not api_key:
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
        
        # 핸들러 초기화
        self.chat_handler = ChatHandler(config)
        self.rag_handler = RAGHandler(config)
        self.cache = VectorCache() if use_cache else None
    
    def load_pdf(self, pdf_path: Union[Path, str]):
        """단일 PDF 로드"""
        pdf_path = Path(pdf_path)
        console.print(f"[cyan]📄 PDF 로딩: {pdf_path.name}[/cyan]")
        
        chunks_count = self.rag_handler.process_pdf(pdf_path)
        console.print(f"[green]✅ 로드 완료: {chunks_count}개 청크[/green]")
        
        return chunks_count
    
    def load_multiple_pdfs(self, pdf_paths: List[Union[Path, str]]):
        """여러 PDF 동시 로드"""
        paths = [Path(p) for p in pdf_paths]
        
        console.print(f"[cyan]📚 {len(paths)}개 PDF 로딩 중...[/cyan]")
        
        results = self.rag_handler.process_multiple_pdfs(paths)
        
        # 결과 출력
        for success in results["success"]:
            console.print(f"[green]✅ {success['file']}: {success['chunks']}개 청크[/green]")
        
        for failed in results["failed"]:
            console.print(f"[red]❌ {failed['file']}: {failed['error']}[/red]")
        
        console.print(f"[cyan]총 {results['total_chunks']}개 청크 로드됨[/cyan]")
        
        return results
    
    def clear_documents(self):
        """로드된 문서 초기화"""
        self.rag_handler.clear_vectorstore()
        console.print("[yellow]문서가 초기화되었습니다.[/yellow]")
    
    def get_loaded_pdfs(self) -> List[str]:
        """로드된 PDF 목록 반환"""
        return self.rag_handler.get_loaded_pdfs()
    
    def query(self, prompt: str, session_name: Optional[str] = None) -> str:
        """질문 처리"""
        if session_name:
            self.chat_handler.load_session(session_name)
        
        response = self.rag_handler.query(
            prompt, 
            chat_history=self.chat_handler.get_history()
        )
        
        self.chat_handler.add_message("user", prompt)
        self.chat_handler.add_message("assistant", response)
        
        if session_name:
            self.chat_handler.save_session(session_name)
        
        return response
    
    def start_repl(self, session_name: Optional[str] = None):
        """대화형 REPL 모드"""
        console.print("[bold cyan]🤖 RAG-GPT REPL 모드[/bold cyan]")
        console.print("종료: 'exit', 'quit', 또는 Ctrl+D")
        console.print("명령어: !help 로 확인\n")
        
        if session_name:
            self.chat_handler.load_session(session_name)
            console.print(f"[green]세션 '{session_name}' 로드됨[/green]\n")
        
        while True:
            try:
                prompt = Prompt.ask("[bold yellow]>[/bold yellow]")
                
                if prompt.lower() in ['exit', 'quit', '종료']:
                    break
                
                if prompt.startswith("!"):
                    self._handle_command(prompt[1:])
                    continue
                
                response = self.query(prompt, session_name)
                
                console.print("\n[bold green]AI:[/bold green]")
                console.print(Markdown(response))
                console.print("\n" + "-"*50 + "\n")
                
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                console.print(f"[red]오류: {e}[/red]")
        
        console.print("\n[cyan]REPL 모드를 종료합니다.[/cyan]")
    
    def _handle_command(self, command: str):
        """특수 명령어 처리 - 파일명 공백 지원"""
        
        # 명령어와 인자 분리
        command = command.strip()
        
        if not command:
            return
        
        # 첫 번째 공백으로 명령어와 나머지 분리
        if " " in command:
            cmd, args_str = command.split(" ", 1)
            cmd = cmd.lower()
        else:
            cmd = command.lower()
            args_str = ""
        
        if cmd == "help":
            console.print("""
[bold cyan]사용 가능한 명령어:[/bold cyan]
  !pdf "파일명.pdf"                    - 단일 PDF 로드
  !pdfs "파일1.pdf" "파일2.pdf"        - 여러 PDF 동시 로드
  !list                                - 로드된 PDF 목록 표시
  !clear                               - 대화 기록 초기화
  !cleardocs                           - 로드된 문서 초기화
  !model <이름>                        - 모델 변경
  !help                                - 도움말 표시

[yellow]참고: 파일명에 공백이 있으면 따옴표로 감싸세요[/yellow]
  예: !pdf "my document.pdf"
  예: !pdfs "file 1.pdf" "file 2.pdf"
            """)
            
        elif cmd == "clear":
            self.chat_handler.clear_history()
            console.print("[yellow]대화 기록 초기화[/yellow]")
            
        elif cmd == "cleardocs":
            self.clear_documents()
            
        elif cmd == "list":
            pdfs = self.get_loaded_pdfs()
            if pdfs:
                console.print("[cyan]로드된 PDF:[/cyan]")
                for i, pdf in enumerate(pdfs, 1):
                    console.print(f"  {i}. 📄 {pdf}")
            else:
                console.print("[yellow]로드된 PDF가 없습니다.[/yellow]")
                
        elif cmd == "pdf":
            if args_str:
                # 따옴표 처리
                try:
                    files = shlex.split(args_str)
                    if files:
                        self.load_pdf(Path(files[0]))
                except ValueError as e:
                    # 따옴표 없이 시도
                    self.load_pdf(Path(args_str.strip()))
            else:
                console.print("[red]사용법: !pdf \"파일명.pdf\"[/red]")
                
        elif cmd == "pdfs":
            if args_str:
                try:
                    # shlex로 따옴표 처리된 파일명들 파싱
                    files = shlex.split(args_str)
                    if files:
                        pdf_paths = [Path(f) for f in files]
                        self.load_multiple_pdfs(pdf_paths)
                except ValueError as e:
                    console.print(f"[red]파일명 파싱 오류: {e}[/red]")
                    console.print("[yellow]파일명에 공백이 있으면 따옴표로 감싸세요[/yellow]")
            else:
                console.print("[red]사용법: !pdfs \"파일1.pdf\" \"파일2.pdf\"[/red]")
                
        elif cmd == "model":
            if args_str:
                model_name = args_str.strip()
                self.config.set("model", model_name)
                self.rag_handler.setup_llm()
                console.print(f"[green]모델 변경: {model_name}[/green]")
            else:
                console.print("[red]사용법: !model <모델이름>[/red]")
                
        else:
            console.print(f"[red]알 수 없는 명령어: {cmd}[/red]")
            console.print("[yellow]!help 로 도움말 확인[/yellow]")
    
    def list_chats(self):
        """대화 목록 표시"""
        self.chat_handler.list_sessions()
    
    def show_chat(self, session_name: Optional[str]):
        """대화 기록 표시"""
        self.chat_handler.show_session(session_name)
