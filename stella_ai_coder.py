#!/usr/bin/env python3
"""
Stella AI Coder

A local terminal coding agent powered by Ollama + Qwen.
Python 3.10+
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text


APP_NAME = "Stella AI Coder"
APP_VERSION = "1.0.0"
DEFAULT_MODEL = os.getenv("STELLA_MODEL", "qwen2.5-coder:1.5b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

MAX_TOOL_ROUNDS = 12
MAX_FILE_CHARS = 28_000
MAX_COMMAND_OUTPUT = 20_000
MAX_CONTEXT_CHARS = 95_000

console = Console()


SYSTEM_PROMPT = """
You are Stella AI Coder, a local terminal coding agent similar to Codex CLI, Claude Code, and Aider.
You help users build, inspect, edit, debug, run, and ship software projects.

Tool protocol:
- When you need a tool, answer ONLY as strict JSON:
  {"tool": "tool_name", "args": {"key": "value"}}
- When done, answer ONLY as strict JSON:
  {"final": "your helpful answer to the user"}

Operating style:
- The user may write in Russian. Reply in the user's language.
- Inspect before editing. Read files or search first when changing existing code.
- Prefer small, correct edits. After meaningful changes, run relevant checks if available.
- Never pretend that you executed a command or saw a file. Use tools for real state.
- Keep paths relative to the active project root unless the user explicitly changes root.
- Powerful shell commands ask the user for confirmation in the app.
- Do not intentionally destroy user data. If a risky operation is needed, explain why.

Available tools:
- list_dir(path="."): list files and folders.
- tree(path=".", depth=3): show a compact project tree.
- find_files(query, path="."): search file names.
- search_text(pattern, path="."): regex search inside text files.
- read_file(path): read a text file.
- write_file(path, content): create or overwrite a text file.
- append_file(path, content): append text to a file.
- edit_file(path, old, new): replace exact text in a file once.
- make_dir(path): create a directory.
- delete_path(path): delete a file or folder after user confirmation.
- run_command(command, reason=""): run a terminal command in the project directory.
- web_fetch(url): download text from a public http/https URL.
""".strip()


ASCII_ART = r"""
   _____ __       ____             ___    ____
  / ___// /____  / / /___ _       /   |  /  _/
  \__ \/ __/ _ \/ / / __ `/______/ /| |  / /
 ___/ / /_/  __/ / / /_/ /_____/ ___ |_/ /
/____/\__/\___/_/_/\__,_/     /_/  |_/___/

       S T E L L A   A I   C O D E R
""".rstrip()


IGNORED_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".next",
    "dist",
    "build",
    ".stella",
}


@dataclass
class ToolResult:
    ok: bool
    content: str


class StellaAgent:
    def __init__(self, model: str = DEFAULT_MODEL, root: Path | None = None) -> None:
        self.model = model
        self.root = (root or Path.cwd()).resolve()
        self.messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.session_file = self._new_session_file()
        self._log("session_start", {"model": self.model, "root": str(self.root), "version": APP_VERSION})

    def _new_session_file(self) -> Path:
        session_dir = self.root / ".stella" / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return session_dir / f"{stamp}.jsonl"

    def _log(self, event: str, data: dict[str, Any]) -> None:
        try:
            record = {"time": dt.datetime.now().isoformat(timespec="seconds"), "event": event, "data": data}
            with self.session_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def clear(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._log("context_clear", {})

    def set_model(self, model: str) -> None:
        self.model = model
        self._log("model_change", {"model": model})

    def set_root(self, root: Path) -> None:
        self.root = root.resolve()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.session_file = self._new_session_file()
        self._log("root_change", {"root": str(self.root), "model": self.model})

    def chat(self, user_text: str) -> str:
        self._add_message("user", user_text)

        for _ in range(MAX_TOOL_ROUNDS):
            self._compact_context_if_needed()
            assistant_text = self._call_ollama()
            self._add_message("assistant", assistant_text)
            payload = parse_json_object(assistant_text)

            if not payload:
                return assistant_text.strip()
            if "final" in payload:
                return str(payload["final"]).strip()

            tool_name = payload.get("tool")
            args = payload.get("args", {})
            if not isinstance(tool_name, str):
                return "The model returned an invalid tool call."
            if not isinstance(args, dict):
                args = {}

            tool_result = self.run_tool(tool_name, args)
            self._print_tool_result(tool_name, args, tool_result)
            self._log("tool", {"name": tool_name, "args": args, "ok": tool_result.ok, "result": tool_result.content})

            tool_message = json.dumps(
                {"tool": tool_name, "ok": tool_result.ok, "result": tool_result.content},
                ensure_ascii=False,
            )
            self._add_message("user", f"TOOL RESULT:\n{tool_message}")

        return "I stopped after several tool calls to avoid an infinite loop. Tell me to continue if needed."

    def _add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._log("message", {"role": role, "content": content})

    def _compact_context_if_needed(self) -> None:
        total = sum(len(item.get("content", "")) for item in self.messages)
        if total <= MAX_CONTEXT_CHARS or len(self.messages) <= 20:
            return

        system = self.messages[:1]
        recent = self.messages[-18:]
        notice = {
            "role": "user",
            "content": "Context note: older messages were compacted automatically to fit the local model context.",
        }
        self.messages = system + [notice] + recent
        self._log("context_compact", {"kept_messages": len(self.messages)})

    def _call_ollama(self) -> str:
        try:
            with console.status("[bold cyan]Stella is thinking...[/bold cyan]", spinner="dots12"):
                response = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": self.model,
                        "messages": self.messages,
                        "stream": False,
                        "options": {"temperature": 0.12, "num_ctx": 8192},
                    },
                    timeout=240,
                )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Ollama is not running. Start it with `ollama serve`, then run "
                f"`ollama pull {self.model}`."
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError("Ollama timed out. Try again or use a smaller model.") from exc

        if response.status_code == 404:
            raise RuntimeError(f"Model `{self.model}` was not found. Run: ollama pull {self.model}")
        if response.status_code >= 400:
            raise RuntimeError(f"Ollama error {response.status_code}: {response.text[:700]}")

        data = response.json()
        return data.get("message", {}).get("content", "").strip()

    def run_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        tools = {
            "list_dir": self.tool_list_dir,
            "tree": self.tool_tree,
            "find_files": self.tool_find_files,
            "search_text": self.tool_search_text,
            "read_file": self.tool_read_file,
            "write_file": self.tool_write_file,
            "append_file": self.tool_append_file,
            "edit_file": self.tool_edit_file,
            "make_dir": self.tool_make_dir,
            "delete_path": self.tool_delete_path,
            "run_command": self.tool_run_command,
            "web_fetch": self.tool_web_fetch,
        }
        fn = tools.get(name)
        if not fn:
            return ToolResult(False, f"Unknown tool: {name}")
        try:
            return fn(**args)
        except TypeError as exc:
            return ToolResult(False, f"Wrong tool arguments for {name}: {exc}")
        except Exception as exc:
            return ToolResult(False, f"{type(exc).__name__}: {exc}")

    def resolve_path(self, user_path: str | None) -> Path:
        raw = (user_path or ".").strip()
        target = (self.root / raw).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError("Path is outside the active project root. Use /cd to change root.")
        return target

    def tool_list_dir(self, path: str = ".") -> ToolResult:
        target = self.resolve_path(path)
        if not target.exists():
            return ToolResult(False, f"Path does not exist: {path}")
        if not target.is_dir():
            return ToolResult(False, f"Not a directory: {path}")

        lines: list[str] = []
        for item in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            marker = "/" if item.is_dir() else ""
            lines.append(f"{item.relative_to(self.root)}{marker}")
        return ToolResult(True, "\n".join(lines) or "(empty)")

    def tool_tree(self, path: str = ".", depth: int = 3) -> ToolResult:
        target = self.resolve_path(path)
        if not target.exists() or not target.is_dir():
            return ToolResult(False, f"Directory does not exist: {path}")
        depth = max(1, min(int(depth), 7))
        lines = [f"{target.relative_to(self.root) if target != self.root else '.'}/"]
        self._walk_tree(target, lines, "", depth)
        return ToolResult(True, "\n".join(lines))

    def _walk_tree(self, folder: Path, lines: list[str], prefix: str, depth: int) -> None:
        if depth <= 0:
            return
        items = [
            p
            for p in sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            if p.name not in IGNORED_NAMES
        ]
        clipped = items[:120]
        for index, item in enumerate(clipped):
            branch = "`-- " if index == len(clipped) - 1 else "|-- "
            lines.append(f"{prefix}{branch}{item.name}{'/' if item.is_dir() else ''}")
            if item.is_dir():
                next_prefix = prefix + ("    " if branch == "`-- " else "|   ")
                self._walk_tree(item, lines, next_prefix, depth - 1)
        if len(items) > len(clipped):
            lines.append(f"{prefix}`-- ... {len(items) - len(clipped)} more")

    def tool_find_files(self, query: str, path: str = ".") -> ToolResult:
        target = self.resolve_path(path)
        if not target.exists() or not target.is_dir():
            return ToolResult(False, f"Directory does not exist: {path}")
        query_lower = query.lower()
        matches: list[str] = []
        for item in target.rglob("*"):
            if any(part in IGNORED_NAMES for part in item.relative_to(self.root).parts):
                continue
            if query_lower in item.name.lower():
                matches.append(str(item.relative_to(self.root)))
            if len(matches) >= 200:
                break
        return ToolResult(True, "\n".join(matches) or "(no matches)")

    def tool_search_text(self, pattern: str, path: str = ".") -> ToolResult:
        target = self.resolve_path(path)
        if not target.exists():
            return ToolResult(False, f"Path does not exist: {path}")
        try:
            regex = re.compile(pattern, flags=re.IGNORECASE)
        except re.error as exc:
            return ToolResult(False, f"Invalid regex: {exc}")

        files = [target] if target.is_file() else [p for p in target.rglob("*") if p.is_file()]
        hits: list[str] = []
        for file_path in files:
            rel_parts = file_path.relative_to(self.root).parts
            if any(part in IGNORED_NAMES for part in rel_parts):
                continue
            if file_path.stat().st_size > 1_000_000:
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{file_path.relative_to(self.root)}:{line_no}: {line[:220]}")
                    if len(hits) >= 200:
                        return ToolResult(True, "\n".join(hits))
        return ToolResult(True, "\n".join(hits) or "(no matches)")

    def tool_read_file(self, path: str) -> ToolResult:
        target = self.resolve_path(path)
        if not target.exists():
            return ToolResult(False, f"File does not exist: {path}")
        if not target.is_file():
            return ToolResult(False, f"Not a file: {path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS] + "\n\n[truncated]"
        return ToolResult(True, text)

    def tool_write_file(self, path: str, content: str) -> ToolResult:
        target = self.resolve_path(path)
        if target.exists():
            rel = target.relative_to(self.root)
            if not Confirm.ask(f"Overwrite existing file `{rel}`?", default=False):
                return ToolResult(False, "User declined overwrite.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(True, f"Written: {target.relative_to(self.root)} ({len(content)} chars)")

    def tool_append_file(self, path: str, content: str) -> ToolResult:
        target = self.resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return ToolResult(True, f"Appended: {target.relative_to(self.root)} ({len(content)} chars)")

    def tool_edit_file(self, path: str, old: str, new: str) -> ToolResult:
        target = self.resolve_path(path)
        if not target.exists() or not target.is_file():
            return ToolResult(False, f"File does not exist: {path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        if old not in text:
            return ToolResult(False, "Exact text to replace was not found.")
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return ToolResult(True, f"Edited: {target.relative_to(self.root)}")

    def tool_make_dir(self, path: str) -> ToolResult:
        target = self.resolve_path(path)
        target.mkdir(parents=True, exist_ok=True)
        return ToolResult(True, f"Directory ready: {target.relative_to(self.root)}")

    def tool_delete_path(self, path: str) -> ToolResult:
        target = self.resolve_path(path)
        if not target.exists():
            return ToolResult(False, f"Path does not exist: {path}")
        rel = target.relative_to(self.root)
        if not Confirm.ask(f"Delete `{rel}`?", default=False):
            return ToolResult(False, "User declined delete.")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return ToolResult(True, f"Deleted: {rel}")

    def tool_web_fetch(self, url: str) -> ToolResult:
        if not re.match(r"^https?://", url):
            return ToolResult(False, "Only http:// and https:// URLs are supported.")
        response = requests.get(url, timeout=35, headers={"User-Agent": "Stella-AI-Coder/1.0"})
        response.raise_for_status()
        text = response.text
        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS] + "\n\n[truncated]"
        return ToolResult(True, text)

    def tool_run_command(self, command: str, reason: str = "") -> ToolResult:
        risk = classify_command(command)
        if risk == "blocked":
            return ToolResult(False, "Command blocked because it looks destructive.")
        if risk == "confirm":
            console.print(
                Panel(
                    f"{command}\n\nReason: {reason or '(not provided)'}",
                    title="Command approval",
                    border_style="yellow",
                    box=box.ROUNDED,
                )
            )
            if not Confirm.ask("Allow Stella to run this command?", default=False):
                return ToolResult(False, "User declined command.")

        completed = subprocess.run(
            command,
            cwd=self.root,
            shell=True,
            text=True,
            capture_output=True,
            timeout=240,
        )
        output = ""
        if completed.stdout:
            output += completed.stdout
        if completed.stderr:
            output += "\n[stderr]\n" + completed.stderr
        output = output.strip() or "(no output)"
        if len(output) > MAX_COMMAND_OUTPUT:
            output = output[:MAX_COMMAND_OUTPUT] + "\n\n[truncated]"
        return ToolResult(completed.returncode == 0, f"exit_code={completed.returncode}\n{output}")

    def _print_tool_result(self, name: str, args: dict[str, Any], result: ToolResult) -> None:
        arg_text = json.dumps(args, ensure_ascii=False)
        content = result.content[:1600]
        color = "green" if result.ok else "red"
        console.print(
            Panel(
                f"[dim]{escape_rich(arg_text)}[/dim]\n\n{escape_rich(content)}",
                title=f"tool: {name} [{'ok' if result.ok else 'error'}]",
                border_style=color,
                box=box.ROUNDED,
            )
        )


def parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def classify_command(command: str) -> str:
    text = command.strip()
    if not text:
        return "blocked"

    blocked = [
        r"\brm\s+-[^\n]*[rf][^\n]*\s+(/|\*|\.|~)",
        r"\bdel\s+(/s|/q|\*)",
        r"\berase\s+(/s|/q|\*)",
        r"\bformat\b",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bmkfs\b",
        r"\bdd\s+",
        r"\bgit\s+reset\s+--hard\b",
        r"\bgit\s+clean\s+-[^\n]*[fdx]",
        r"\bRemove-Item\b.*\s-Recurse\b",
    ]
    for pattern in blocked:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return "blocked"

    lower = text.lower()
    safe_git = re.match(r"^git\s+(status|diff|log|show|branch|remote|rev-parse|ls-files)\b", lower)
    if safe_git:
        return "auto"

    first = text.split()[0].lower()
    auto_safe = {
        "python",
        "python3",
        "py",
        "pytest",
        "ruff",
        "mypy",
        "node",
        "dir",
        "ls",
        "pwd",
        "echo",
        "type",
        "cat",
        "find",
        "findstr",
        "rg",
        "tree",
        "get-childitem",
        "get-content",
    }
    if first in auto_safe:
        return "auto"

    confirm_needed = {
        "pip",
        "npm",
        "pnpm",
        "yarn",
        "curl",
        "wget",
        "ssh",
        "scp",
        "rsync",
        "docker",
        "docker-compose",
        "gh",
        "git",
        "ollama",
        "powershell",
        "pwsh",
        "cmd",
        "bash",
        "sh",
    }
    if first in confirm_needed:
        return "confirm"
    return "confirm"


def check_ollama(model: str, quiet: bool = False) -> bool:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    except requests.RequestException:
        if not quiet:
            console.print(
                Panel(
                    "Ollama is not responding.\n\n"
                    "Install Ollama: https://ollama.com/download\n"
                    f"Then run: ollama pull {model}",
                    title="Ollama offline",
                    border_style="red",
                )
            )
        return False

    if response.status_code >= 400:
        if not quiet:
            console.print(f"[red]Ollama error:[/red] {response.status_code}")
        return False

    tags = response.json().get("models", [])
    names = {item.get("name") for item in tags}
    if model not in names:
        if not quiet:
            console.print(
                Panel(
                    f"Model `{model}` is not installed yet.\n\nRun: ollama pull {model}",
                    title="Model missing",
                    border_style="yellow",
                )
            )
        return False

    return True


def print_banner(model: str, root: Path) -> None:
    console.print(Text(ASCII_ART, style="bold bright_cyan"))
    console.print(
        Panel(
            f"[bold magenta]{APP_NAME}[/bold magenta] [dim]v{APP_VERSION}[/dim]\n"
            f"[cyan]model[/cyan]: {model}\n"
            f"[cyan]project[/cyan]: {root}\n\n"
            "Chat normally. Commands: [bold]/help[/bold], [bold]/doctor[/bold], [bold]/model[/bold], "
            "[bold]/cd[/bold], [bold]/clear[/bold], [bold]/exit[/bold]",
            border_style="bright_magenta",
            box=box.DOUBLE,
        )
    )


def print_help() -> None:
    table = Table(title="Stella AI Coder", box=box.ROUNDED, border_style="cyan")
    table.add_column("Command / tool", style="bold magenta")
    table.add_column("What it does", style="white")
    table.add_row("/help", "Show help")
    table.add_row("/doctor", "Check Python, Ollama, Git, GitHub CLI, and current model")
    table.add_row("/model NAME", "Switch Ollama model, for example /model qwen2.5-coder:3b")
    table.add_row("/cd PATH", "Change active project root")
    table.add_row("/pwd", "Show active project root")
    table.add_row("/clear", "Clear chat memory for this session")
    table.add_row("/exit", "Exit")
    table.add_row("tree / list_dir / find_files / search_text", "Inspect project structure and code")
    table.add_row("read_file / write_file / append_file / edit_file", "Read and edit files")
    table.add_row("make_dir / delete_path", "Create or delete paths; delete asks for confirmation")
    table.add_row("run_command", "Run terminal commands; powerful commands ask for confirmation")
    table.add_row("web_fetch", "Read public URLs such as GitHub raw files or docs")
    console.print(table)


def print_doctor(model: str) -> None:
    table = Table(title="Stella Doctor", box=box.ROUNDED, border_style="cyan")
    table.add_column("Check", style="bold magenta")
    table.add_column("Result", style="white")
    table.add_row("Python", sys.version.split()[0])
    table.add_row("Ollama API", "ok" if requests_ok(f"{OLLAMA_URL}/api/tags") else "not responding")
    table.add_row("Model", f"{model} ok" if check_ollama(model, quiet=True) else f"{model} missing/offline")
    table.add_row("git", shutil.which("git") or "not found")
    table.add_row("gh", shutil.which("gh") or "not found")
    table.add_row("docker", shutil.which("docker") or "not found")
    table.add_row("node", shutil.which("node") or "not found")
    table.add_row("npm", shutil.which("npm") or "not found")
    console.print(table)


def requests_ok(url: str) -> bool:
    try:
        response = requests.get(url, timeout=4)
        return response.status_code < 500
    except requests.RequestException:
        return False


def render_answer(text: str) -> None:
    if "```" in text or text.lstrip().startswith(("#", "-", "1.")):
        console.print(Panel(Markdown(text), title="Stella", border_style="bright_cyan", box=box.ROUNDED))
    else:
        console.print(Panel(escape_rich(text), title="Stella", border_style="bright_cyan", box=box.ROUNDED))


def escape_rich(text: str) -> str:
    return text.replace("[", "\\[").replace("]", "\\]")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stella AI Coder")
    parser.add_argument("model_arg", nargs="?", help="Optional Ollama model name")
    parser.add_argument("--model", dest="model", help="Ollama model name")
    parser.add_argument("--root", dest="root", help="Project root")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.version:
        console.print(f"{APP_NAME} {APP_VERSION}")
        return 0

    model = args.model or args.model_arg or DEFAULT_MODEL
    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    print_banner(model, root)

    if not check_ollama(model):
        return 1

    agent = StellaAgent(model=model, root=root)
    print_help()

    while True:
        try:
            user_text = Prompt.ask("\n[bold bright_green]you[/bold bright_green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[magenta]Bye. Stella is shutting down.[/magenta]")
            return 0

        if not user_text:
            continue

        lower = user_text.lower()
        if lower in {"/exit", "exit", "quit"}:
            console.print("[magenta]Bye. Stella is shutting down.[/magenta]")
            return 0
        if lower == "/help":
            print_help()
            continue
        if lower == "/doctor":
            print_doctor(agent.model)
            continue
        if lower == "/pwd":
            console.print(Panel(str(agent.root), title="Project root", border_style="cyan"))
            continue
        if lower == "/clear":
            agent.clear()
            console.print("[green]Context cleared.[/green]")
            continue
        if lower.startswith("/model"):
            parts = user_text.split(maxsplit=1)
            if len(parts) == 1:
                console.print(Panel(agent.model, title="Current model", border_style="cyan"))
                continue
            new_model = parts[1].strip()
            agent.set_model(new_model)
            if check_ollama(new_model):
                console.print(f"[green]Model switched to {new_model}[/green]")
            continue
        if lower.startswith("/cd"):
            parts = user_text.split(maxsplit=1)
            if len(parts) == 1:
                console.print("[yellow]Usage: /cd PATH[/yellow]")
                continue
            new_root = Path(parts[1]).expanduser().resolve()
            if not new_root.exists() or not new_root.is_dir():
                console.print(f"[red]Directory not found:[/red] {new_root}")
                continue
            agent.set_root(new_root)
            console.print(f"[green]Project root changed:[/green] {new_root}")
            continue

        try:
            answer = agent.chat(user_text)
        except RuntimeError as exc:
            console.print(Panel(str(exc), title="Error", border_style="red"))
            continue
        except Exception as exc:
            console.print(Panel(f"{type(exc).__name__}: {exc}", title="Unexpected error", border_style="red"))
            continue

        render_answer(answer)


if __name__ == "__main__":
    raise SystemExit(main())
