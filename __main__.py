#!/usr/bin/env python
"""
RAG-GPT: shell_gpt 스타일의 RAG CLI 도구
"""
import sys
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console

from .app import RagGPT
from .config import Config

console = Console()
app = typer.Typer(
    help="RAG 기반 문서 질의응답 CLI",
    add_completion=False,
    rich_markup_mode="rich"
)

@app.command()
def main(
    prompt: Optional[str] = typer.Argument(None, help="질문 또는 프롬프트"),
    pdf: Optional[Path] = typer.Option(None, "--pdf", "-p", help="PDF 파일 경로"),
    chat: Optional[str] = typer.Option(None, "--chat", "-c", help="대화 세션 이름"),
    repl: bool = typer.Option(False, "--repl", "-r", help="대화형 REPL 모드"),
    web: bool = typer.Option(False, "--web", "-w", help="웹 인터페이스 실행"),
    port: int = typer.Option(7860, "--port", help="웹 서버 포트"),
    share: bool = typer.Option(False, "--share", help="공개 URL 생성 (ngrok)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="사용할 LLM 모델"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Temperature"),
    no_cache: bool = typer.Option(False, "--no-cache", help="캐시 사용 안 함"),
    show_chat: bool = typer.Option(False, "--show-chat", "-s", help="대화 기록 표시"),
    list_chats: bool = typer.Option(False, "--list-chats", "-l", help="모든 대화 목록"),
):
    """
    RAG-GPT: 문서 기반 AI 어시스턴트
    
    Examples:
        # 웹 인터페이스 실행
        rag-gpt --web
        
        # 공개 URL로 웹 실행
        rag-gpt --web --share
        
        # CLI 모드
        rag-gpt --repl --pdf document.pdf
    """
    
    # 설정 초기화
    config = Config()
    
    # 모델 설정
    if model:
        config.set("model", model)
    if temperature is not None:
        config.set("temperature", temperature)
    
    # RagGPT 인스턴스 생성
    try:
        rag_gpt = RagGPT(config, use_cache=not no_cache)
    except ValueError as e:
        console.print(f"[red]오류: {e}[/red]")
        sys.exit(1)
    
    # 웹 모드
    if web:
        console.print(f"[cyan]🌐 웹 인터페이스 시작 (포트: {port})[/cyan]")
        from .web_app import WebInterface
        
        web_ui = WebInterface(rag_gpt)
        web_ui.launch(
            server_port=port,
            server_name="0.0.0.0",
            share=share,
            inbrowser=not share  # share 모드가 아닐 때만 브라우저 자동 열기
        )
        return
        
    # 대화 목록 표시
    if list_chats:
        rag_gpt.list_chats()
        return
    
    # 대화 기록 표시
    if show_chat:
        rag_gpt.show_chat(chat)
        return
    
    # PDF 로드
    if pdf:
        rag_gpt.load_pdf(pdf)
    
    # REPL 모드
    if repl:
        rag_gpt.start_repl(session_name=chat)
        return
    
    # 단일 프롬프트 처리
    if prompt:
        response = rag_gpt.query(prompt, session_name=chat)
        console.print(response)
    else:
        console.print("[yellow]프롬프트를 입력하거나 --repl 옵션을 사용하세요.[/yellow]")

if __name__ == "__main__":
    app()

