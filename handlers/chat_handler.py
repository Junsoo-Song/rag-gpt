"""
대화 처리 핸들러
"""
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from rich.console import Console
from rich.table import Table
from langchain_core.messages import HumanMessage, AIMessage

console = Console()

class ChatHandler:
    """대화 처리 핸들러"""
    
    def __init__(self, config):
        self.config = config
        self.sessions_dir = Path.home() / ".rag_gpt" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.current_history = []
    
    def add_message(self, role: str, content: str):
        """메시지 추가"""
        if role == "user":
            self.current_history.append(HumanMessage(content=content))
        elif role == "assistant":
            self.current_history.append(AIMessage(content=content))
    
    def get_history(self) -> List:
        """현재 대화 기록 반환"""
        return self.current_history
    
    def clear_history(self):
        """대화 기록 초기화"""
        self.current_history = []
    
    def save_session(self, name: str):
        """세션 저장"""
        session_file = self.sessions_dir / f"{name}.json"
        data = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "messages": [
                {
                    "role": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": msg.content
                }
                for msg in self.current_history
            ]
        }
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_session(self, name: str):
        """세션 로드"""
        session_file = self.sessions_dir / f"{name}.json"
        if session_file.exists():
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.current_history = []
            for msg in data.get("messages", []):
                if msg["role"] == "human":
                    self.current_history.append(HumanMessage(content=msg["content"]))
                else:
                    self.current_history.append(AIMessage(content=msg["content"]))
    
    def list_sessions(self):
        """모든 세션 목록 표시"""
        sessions = list(self.sessions_dir.glob("*.json"))
        
        if not sessions:
            console.print("[yellow]저장된 세션이 없습니다.[/yellow]")
            return
        
        table = Table(title="저장된 세션")
        table.add_column("이름", style="cyan")
        table.add_column("날짜", style="green")
        table.add_column("메시지 수", style="yellow")
        
        for session_file in sessions:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            table.add_row(
                data["name"],
                data.get("timestamp", "Unknown")[:10],
                str(len(data.get("messages", [])))
            )
        
        console.print(table)
    
    def show_session(self, name: Optional[str]):
        """세션 내용 표시"""
        if not name:
            console.print("[yellow]세션 이름을 지정하세요.[/yellow]")
            return
        
        session_file = self.sessions_dir / f"{name}.json"
        if not session_file.exists():
            console.print(f"[red]세션 '{name}'을 찾을 수 없습니다.[/red]")
            return
        
        with open(session_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        console.print(f"\n[bold cyan]세션: {name}[/bold cyan]")
        console.print(f"[dim]날짜: {data.get('timestamp', 'Unknown')[:10]}[/dim]\n")
        
        for msg in data.get("messages", []):
            if msg["role"] == "human":
                console.print(f"[bold yellow]User:[/bold yellow] {msg['content']}")
            else:
                console.print(f"[bold green]AI:[/bold green] {msg['content']}")
            console.print()
    
    def show_history(self):
        """현재 대화 기록 표시"""
        if not self.current_history:
            console.print("[yellow]대화 기록이 없습니다.[/yellow]")
            return
        
        for msg in self.current_history:
            if isinstance(msg, HumanMessage):
                console.print(f"[bold yellow]User:[/bold yellow] {msg.content}")
            else:
                console.print(f"[bold green]AI:[/bold green] {msg.content}")
            console.print()

