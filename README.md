# Stella AI Coder

Beautiful local terminal AI coding agent powered by Ollama and Qwen.

Stella can inspect files, edit code, create projects, run tests, use Git/GitHub CLI, fetch public URLs, and run powerful terminal commands after user confirmation.

## One-command install

Replace `YOUR_GITHUB_USERNAME` with your GitHub username after publishing this repo.

### Windows PowerShell

```powershell
irm https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/stella-ai-coder/main/install.ps1 | iex
```

After installation:

```powershell
stella
```

### macOS / Linux

```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/stella-ai-coder/main/install.sh | sh
```

After installation:

```bash
stella
```

## Publish to your GitHub

On Windows PowerShell, from this folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\publish_to_github.ps1
```

It will install/use GitHub CLI, open GitHub login if needed, create `stella-ai-coder`, push files, and print the final install command for friends.

## Stronger model

Default model:

```text
qwen2.5-coder:1.5b
```

Use a stronger model:

```powershell
$env:STELLA_MODEL="qwen2.5-coder:3b"; irm https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/stella-ai-coder/main/install.ps1 | iex
```

or after install:

```text
/model qwen2.5-coder:3b
```

## Chat commands

- `/help` shows tools and commands
- `/doctor` checks Python, Ollama, Git, GitHub CLI, Docker, Node, npm
- `/model NAME` switches Ollama model
- `/cd PATH` changes active project root
- `/pwd` shows active project root
- `/clear` clears session context
- `/exit` quits

## Tool safety

Stella can run real terminal commands, including `git`, `gh`, `ssh`, `docker`, `npm`, and `pip`.
Commands that can change the system ask for confirmation first. Obviously destructive commands are blocked.
