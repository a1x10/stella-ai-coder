param(
    [string]$RepoName = "stella-ai-coder",
    [switch]$Private
)

$ErrorActionPreference = "Stop"

$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")

function Test-Command($Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Run-Git($ArgsList) {
    & git @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "git $($ArgsList -join ' ') failed"
    }
}

if (-not (Test-Command "git")) {
    throw "Git is required. Install Git for Windows and run again."
}

if (-not (Test-Command "gh")) {
    Write-Host "GitHub CLI was not found." -ForegroundColor Yellow
    if (Test-Command "winget") {
        $ok = Read-Host "Install GitHub CLI with winget? Type Y to continue"
        if ($ok -match "^[Yy]$") {
            winget install -e --id GitHub.cli
        } else {
            throw "GitHub CLI is required to publish automatically."
        }
    } else {
        throw "Install GitHub CLI from https://cli.github.com and run again."
    }
}

gh auth status | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Opening GitHub login in browser..." -ForegroundColor Cyan
    gh auth login --hostname github.com --web --git-protocol https --scopes repo
    if ($LASTEXITCODE -ne 0) {
        throw "GitHub login failed."
    }
}

$owner = gh api user --jq .login
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($owner)) {
    throw "Could not detect GitHub username."
}

$raw = "https://raw.githubusercontent.com/$owner/$RepoName/main"
$placeholder = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/stella-ai-coder/main"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

foreach ($file in @("install.ps1", "install.sh", "README.md")) {
    $text = Get-Content $file -Raw
    $text = $text.Replace($placeholder, $raw)
    $text = $text.Replace("YOUR_GITHUB_USERNAME", $owner)
    [System.IO.File]::WriteAllText((Resolve-Path $file), $text, $utf8NoBom)
}

if (-not (Test-Path ".git")) {
    Run-Git @("init")
}

Run-Git @("branch", "-M", "main")
Run-Git @("add", "stella_ai_coder.py", "requirements.txt", "install.ps1", "install.sh", "README.md", "publish_to_github.ps1", ".gitignore")

git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "No new changes to commit." -ForegroundColor DarkGray
} else {
    git commit -m "Initial Stella AI Coder release"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Git commit failed. If Git asks for name/email, run:" -ForegroundColor Yellow
        Write-Host '  git config --global user.name "Your Name"'
        Write-Host '  git config --global user.email "you@example.com"'
        throw "Commit failed."
    }
}

$visibility = if ($Private) { "--private" } else { "--public" }

gh repo view "$owner/$RepoName" | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Repository already exists: $owner/$RepoName" -ForegroundColor Yellow
    git remote remove origin 2>$null
    Run-Git @("remote", "add", "origin", "https://github.com/$owner/$RepoName.git")
    Run-Git @("push", "-u", "origin", "main")
} else {
    gh repo create $RepoName $visibility --source . --remote origin --push --description "Local terminal AI coding agent powered by Ollama and Qwen"
    if ($LASTEXITCODE -ne 0) {
        throw "gh repo create failed."
    }
}

Write-Host ""
Write-Host "Published:" -ForegroundColor Green
Write-Host "https://github.com/$owner/$RepoName"
Write-Host ""
Write-Host "Command for friends:" -ForegroundColor Cyan
Write-Host "irm $raw/install.ps1 | iex" -ForegroundColor White
