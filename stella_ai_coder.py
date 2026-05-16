#!/usr/bin/env python3
"""
Stella AI Coder

A local terminal coding agent powered by Ollama + Qwen.
Python 3.10+
"""

from __future__ import annotations

import argparse
import datetime as dt
import html as html_lib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import webbrowser
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
APP_VERSION = "1.1.0"
DEFAULT_MODEL = os.getenv("STELLA_MODEL", "qwen2.5-coder:1.5b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

MAX_TOOL_ROUNDS = 12
MAX_FILE_CHARS = 28_000
MAX_COMMAND_OUTPUT = 20_000
MAX_CONTEXT_CHARS = 95_000

console = Console()


SYSTEM_PROMPT = """
Ты Stella AI Coder, локальный терминальный ИИ-агент для программирования, похожий на Codex CLI, Claude Code и Aider.
Ты помогаешь пользователю создавать проекты, читать и редактировать код, запускать тесты, работать с Git/GitHub,
искать информацию в интернете, открывать браузер, настраивать серверы и собирать готовые решения.

Протокол инструментов:
- Когда нужен инструмент, отвечай ТОЛЬКО строгим JSON:
  {"tool": "tool_name", "args": {"key": "value"}}
- Когда задача завершена, отвечай ТОЛЬКО строгим JSON:
  {"final": "полезный ответ пользователю на русском"}

Стиль работы:
- Всегда отвечай по-русски, если пользователь явно не попросил другой язык.
- Действуй как аккуратный senior-разработчик: сначала изучай файлы, потом меняй.
- Если нужно написать проект, создай файлы, установи зависимости через команды, запусти тест/проверку и объясни результат.
- Никогда не притворяйся, что видел файл, сайт или вывод команды. Для реального состояния используй инструменты.
- Пути держи внутри активной папки проекта, если пользователь явно не сменил корень через /папка или /cd.
- Команды, которые меняют систему, сеть, серверы, GitHub, Docker, npm/pip/ssh, требуют подтверждения пользователя.
- Не делай скрытое удалённое управление, обход авторизации, кражу токенов, вредоносный код или разрушительные действия.
- Если пользователь просит Telegram-бота для управления своим ПК, делай только легальный вариант: явное согласие,
  токен в .env, allowlist команд, логирование, подтверждения опасных действий, без скрытности и автозапуска без согласия.

Инструменты:
- list_dir(path="."): список файлов и папок.
- tree(path=".", depth=3): компактное дерево проекта.
- find_files(query, path="."): поиск файлов по имени.
- search_text(pattern, path="."): поиск текста/regex внутри файлов.
- read_file(path): чтение текстового файла.
- write_file(path, content): создание или перезапись файла.
- append_file(path, content): добавление текста в файл.
- edit_file(path, old, new): точная замена фрагмента в файле.
- make_dir(path): создание папки.
- delete_path(path): удаление файла/папки после подтверждения.
- run_command(command, reason=""): запуск терминальной команды в папке проекта.
- web_search(query, max_results=5): поиск в интернете.
- web_fetch(url): чтение публичного URL.
- open_url(url): открыть ссылку в браузере пользователя.
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
                return "Модель вернула некорректный вызов инструмента."
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

        return "Я остановилась после нескольких вызовов инструментов, чтобы не уйти в бесконечный цикл. Напиши `продолжай`, если нужно идти дальше."

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
            with console.status("[bold cyan]Stella думает...[/bold cyan]", spinner="dots12"):
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
                "Ollama не запущена. Запусти `ollama serve`, затем выполни "
                f"`ollama pull {self.model}`."
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError("Ollama слишком долго не отвечает. Повтори запрос или выбери модель поменьше.") from exc

        if response.status_code == 404:
            raise RuntimeError(f"Модель `{self.model}` не найдена. Выполни: ollama pull {self.model}")
        if response.status_code >= 400:
            raise RuntimeError(f"Ошибка Ollama {response.status_code}: {response.text[:700]}")

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
            "web_search": self.tool_web_search,
            "web_fetch": self.tool_web_fetch,
            "open_url": self.tool_open_url,
        }
        fn = tools.get(name)
        if not fn:
            return ToolResult(False, f"Неизвестный инструмент: {name}")
        try:
            return fn(**args)
        except TypeError as exc:
            return ToolResult(False, f"Неверные аргументы инструмента {name}: {exc}")
        except Exception as exc:
            return ToolResult(False, f"{type(exc).__name__}: {exc}")

    def resolve_path(self, user_path: str | None) -> Path:
        raw = (user_path or ".").strip()
        target = (self.root / raw).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError("Путь находится вне активной папки проекта. Используй /папка или /cd, чтобы сменить корень.")
        return target

    def tool_list_dir(self, path: str = ".") -> ToolResult:
        target = self.resolve_path(path)
        if not target.exists():
            return ToolResult(False, f"Путь не существует: {path}")
        if not target.is_dir():
            return ToolResult(False, f"Это не папка: {path}")

        lines: list[str] = []
        for item in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            marker = "/" if item.is_dir() else ""
            lines.append(f"{item.relative_to(self.root)}{marker}")
        return ToolResult(True, "\n".join(lines) or "(пусто)")

    def tool_tree(self, path: str = ".", depth: int = 3) -> ToolResult:
        target = self.resolve_path(path)
        if not target.exists() or not target.is_dir():
            return ToolResult(False, f"Папка не существует: {path}")
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
            return ToolResult(False, f"Папка не существует: {path}")
        query_lower = query.lower()
        matches: list[str] = []
        for item in target.rglob("*"):
            if any(part in IGNORED_NAMES for part in item.relative_to(self.root).parts):
                continue
            if query_lower in item.name.lower():
                matches.append(str(item.relative_to(self.root)))
            if len(matches) >= 200:
                break
        return ToolResult(True, "\n".join(matches) or "(ничего не найдено)")

    def tool_search_text(self, pattern: str, path: str = ".") -> ToolResult:
        target = self.resolve_path(path)
        if not target.exists():
            return ToolResult(False, f"Путь не существует: {path}")
        try:
            regex = re.compile(pattern, flags=re.IGNORECASE)
        except re.error as exc:
            return ToolResult(False, f"Некорректный regex: {exc}")

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
        return ToolResult(True, "\n".join(hits) or "(ничего не найдено)")

    def tool_read_file(self, path: str) -> ToolResult:
        target = self.resolve_path(path)
        if not target.exists():
            return ToolResult(False, f"Файл не существует: {path}")
        if not target.is_file():
            return ToolResult(False, f"Это не файл: {path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS] + "\n\n[обрезано]"
        return ToolResult(True, text)

    def tool_write_file(self, path: str, content: str) -> ToolResult:
        target = self.resolve_path(path)
        if target.exists():
            rel = target.relative_to(self.root)
            if not Confirm.ask(f"Перезаписать существующий файл `{rel}`?", default=False):
                return ToolResult(False, "Пользователь отказался от перезаписи.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(True, f"Записано: {target.relative_to(self.root)} ({len(content)} символов)")

    def tool_append_file(self, path: str, content: str) -> ToolResult:
        target = self.resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return ToolResult(True, f"Добавлено: {target.relative_to(self.root)} ({len(content)} символов)")

    def tool_edit_file(self, path: str, old: str, new: str) -> ToolResult:
        target = self.resolve_path(path)
        if not target.exists() or not target.is_file():
            return ToolResult(False, f"Файл не существует: {path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        if old not in text:
            return ToolResult(False, "Точный фрагмент для замены не найден.")
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return ToolResult(True, f"Изменено: {target.relative_to(self.root)}")

    def tool_make_dir(self, path: str) -> ToolResult:
        target = self.resolve_path(path)
        target.mkdir(parents=True, exist_ok=True)
        return ToolResult(True, f"Папка готова: {target.relative_to(self.root)}")

    def tool_delete_path(self, path: str) -> ToolResult:
        target = self.resolve_path(path)
        if not target.exists():
            return ToolResult(False, f"Путь не существует: {path}")
        rel = target.relative_to(self.root)
        if not Confirm.ask(f"Удалить `{rel}`?", default=False):
            return ToolResult(False, "Пользователь отказался от удаления.")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return ToolResult(True, f"Удалено: {rel}")

    def tool_web_search(self, query: str, max_results: int = 5) -> ToolResult:
        max_results = max(1, min(int(max_results), 10))
        url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        response = requests.get(
            url,
            timeout=35,
            headers={"User-Agent": "Mozilla/5.0 Stella-AI-Coder/1.1"},
        )
        response.raise_for_status()

        results: list[str] = []
        blocks = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', response.text, re.I | re.S)
        for raw_href, raw_title in blocks:
            href = html_lib.unescape(raw_href)
            parsed = urllib.parse.urlparse(href)
            query_params = urllib.parse.parse_qs(parsed.query)
            if "uddg" in query_params:
                href = query_params["uddg"][0]
            title = strip_html(raw_title)
            if title and href:
                results.append(f"- {title}\n  {href}")
            if len(results) >= max_results:
                break

        if not results:
            return ToolResult(False, "Ничего не найдено. Попробуй другой запрос или дай прямую ссылку для web_fetch.")
        return ToolResult(True, "\n".join(results))

    def tool_web_fetch(self, url: str) -> ToolResult:
        if not re.match(r"^https?://", url):
            return ToolResult(False, "Поддерживаются только ссылки http:// и https://.")
        response = requests.get(url, timeout=35, headers={"User-Agent": "Stella-AI-Coder/1.1"})
        response.raise_for_status()
        text = response.text
        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS] + "\n\n[обрезано]"
        return ToolResult(True, text)

    def tool_open_url(self, url: str) -> ToolResult:
        if not re.match(r"^https?://", url):
            return ToolResult(False, "Поддерживаются только ссылки http:// и https://.")
        if not Confirm.ask(f"Открыть в браузере: {url}?", default=True):
            return ToolResult(False, "Пользователь отказался открыть браузер.")
        opened = webbrowser.open(url)
        return ToolResult(opened, "Браузер открыт." if opened else "Не удалось открыть браузер.")

    def tool_run_command(self, command: str, reason: str = "") -> ToolResult:
        risk = classify_command(command)
        if risk == "blocked":
            return ToolResult(False, "Команда заблокирована: она выглядит разрушительной.")
        if risk == "confirm":
            console.print(
                Panel(
                    f"{command}\n\nПричина: {reason or '(не указана)'}",
                    title="Подтверждение команды",
                    border_style="yellow",
                    box=box.ROUNDED,
                )
            )
            if not Confirm.ask("Разрешить Stella выполнить эту команду?", default=False):
                return ToolResult(False, "Пользователь отказался выполнить команду.")

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
        output = output.strip() or "(нет вывода)"
        if len(output) > MAX_COMMAND_OUTPUT:
            output = output[:MAX_COMMAND_OUTPUT] + "\n\n[обрезано]"
        return ToolResult(completed.returncode == 0, f"exit_code={completed.returncode}\n{output}")

    def _print_tool_result(self, name: str, args: dict[str, Any], result: ToolResult) -> None:
        arg_text = json.dumps(args, ensure_ascii=False)
        content = result.content[:1600]
        color = "green" if result.ok else "red"
        console.print(
            Panel(
                f"[dim]{escape_rich(arg_text)}[/dim]\n\n{escape_rich(content)}",
                title=f"инструмент: {name} [{'готово' if result.ok else 'ошибка'}]",
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
                    "Ollama не отвечает.\n\n"
                    "Установи Ollama: https://ollama.com/download\n"
                    f"Затем выполни: ollama pull {model}",
                    title="Ollama не запущена",
                    border_style="red",
                )
            )
        return False

    if response.status_code >= 400:
        if not quiet:
            console.print(f"[red]Ошибка Ollama:[/red] {response.status_code}")
        return False

    tags = response.json().get("models", [])
    names = {item.get("name") for item in tags}
    if model not in names:
        if not quiet:
            console.print(
                Panel(
                    f"Модель `{model}` ещё не установлена.\n\nВыполни: ollama pull {model}",
                    title="Модель не найдена",
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
            f"[cyan]модель[/cyan]: {model}\n"
            f"[cyan]проект[/cyan]: {root}\n\n"
            "Пиши обычным языком. Команды: [bold]/помощь[/bold], [bold]/доктор[/bold], [bold]/модель[/bold], "
            "[bold]/папка[/bold], [bold]/очистить[/bold], [bold]/выход[/bold]",
            border_style="bright_magenta",
            box=box.DOUBLE,
        )
    )


def print_help() -> None:
    table = Table(title="Stella AI Coder", box=box.ROUNDED, border_style="cyan")
    table.add_column("Команда / инструмент", style="bold magenta")
    table.add_column("Что делает", style="white")
    table.add_row("/помощь или /help", "Показать справку")
    table.add_row("/доктор или /doctor", "Проверить Python, Ollama, Git, GitHub CLI, Docker, Node, npm")
    table.add_row("/модель NAME", "Переключить модель, например /модель qwen2.5-coder:3b")
    table.add_row("/папка PATH", "Сменить активную папку проекта")
    table.add_row("/где или /pwd", "Показать активную папку проекта")
    table.add_row("/очистить", "Очистить память текущей сессии")
    table.add_row("/выход", "Выйти")
    table.add_row("tree / list_dir / find_files / search_text", "Изучение структуры проекта и кода")
    table.add_row("read_file / write_file / append_file / edit_file", "Чтение и редактирование файлов")
    table.add_row("make_dir / delete_path", "Создание и удаление; удаление требует подтверждения")
    table.add_row("run_command", "Запуск команд; мощные команды требуют подтверждения")
    table.add_row("web_search / web_fetch / open_url", "Поиск в интернете, чтение URL и открытие браузера")
    console.print(table)


def print_doctor(model: str) -> None:
    table = Table(title="Диагностика Stella", box=box.ROUNDED, border_style="cyan")
    table.add_column("Проверка", style="bold magenta")
    table.add_column("Результат", style="white")
    table.add_row("Python", sys.version.split()[0])
    table.add_row("Ollama API", "готово" if requests_ok(f"{OLLAMA_URL}/api/tags") else "не отвечает")
    table.add_row("Модель", f"{model} готова" if check_ollama(model, quiet=True) else f"{model} не найдена/offline")
    table.add_row("git", shutil.which("git") or "не найден")
    table.add_row("gh", shutil.which("gh") or "не найден")
    table.add_row("docker", shutil.which("docker") or "не найден")
    table.add_row("node", shutil.which("node") or "не найден")
    table.add_row("npm", shutil.which("npm") or "не найден")
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


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return html_lib.unescape(re.sub(r"\s+", " ", text)).strip()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stella AI Coder")
    parser.add_argument("model_arg", nargs="?", help="Необязательное имя модели Ollama")
    parser.add_argument("--model", dest="model", help="Имя модели Ollama")
    parser.add_argument("--root", dest="root", help="Папка проекта")
    parser.add_argument("--version", action="store_true", help="Показать версию и выйти")
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
            user_text = Prompt.ask("\n[bold bright_green]ты[/bold bright_green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[magenta]Пока. Stella выключается.[/magenta]")
            return 0

        if not user_text:
            continue

        lower = user_text.lower()
        if lower in {"/exit", "exit", "quit", "/выход", "выход"}:
            console.print("[magenta]Пока. Stella выключается.[/magenta]")
            return 0
        if lower in {"/help", "/помощь", "/команды"}:
            print_help()
            continue
        if lower in {"/doctor", "/доктор", "/диагностика"}:
            print_doctor(agent.model)
            continue
        if lower in {"/pwd", "/где", "/папка?"}:
            console.print(Panel(str(agent.root), title="Папка проекта", border_style="cyan"))
            continue
        if lower in {"/clear", "/очистить"}:
            agent.clear()
            console.print("[green]Контекст очищен.[/green]")
            continue
        if lower.startswith("/model") or lower.startswith("/модель"):
            parts = user_text.split(maxsplit=1)
            if len(parts) == 1:
                console.print(Panel(agent.model, title="Текущая модель", border_style="cyan"))
                continue
            new_model = parts[1].strip()
            agent.set_model(new_model)
            if check_ollama(new_model):
                console.print(f"[green]Модель переключена на {new_model}[/green]")
            continue
        if lower.startswith("/cd") or lower.startswith("/папка"):
            parts = user_text.split(maxsplit=1)
            if len(parts) == 1:
                console.print("[yellow]Использование: /папка PATH[/yellow]")
                continue
            new_root = Path(parts[1]).expanduser().resolve()
            if not new_root.exists() or not new_root.is_dir():
                console.print(f"[red]Папка не найдена:[/red] {new_root}")
                continue
            agent.set_root(new_root)
            console.print(f"[green]Папка проекта изменена:[/green] {new_root}")
            continue

        try:
            answer = agent.chat(user_text)
        except RuntimeError as exc:
            console.print(Panel(str(exc), title="Ошибка", border_style="red"))
            continue
        except Exception as exc:
            console.print(Panel(f"{type(exc).__name__}: {exc}", title="Неожиданная ошибка", border_style="red"))
            continue

        render_answer(answer)


if __name__ == "__main__":
    raise SystemExit(main())
