param(
    [Parameter(Mandatory = $false)]
    [ValidatePattern('^\d{4}-\d{2}-\d{2}$')]
    [string]$Date = (Get-Date -Format 'yyyy-MM-dd'),

    [Parameter(Mandatory = $false)]
    [switch]$IncludeWorkingTree
)

$ErrorActionPreference = 'Stop'

$repoRoot = git rev-parse --show-toplevel 2>$null
if (-not $repoRoot) {
    throw 'Run this script inside a Git repository.'
}

$start = "$Date 00:00:00"
$end = "$Date 23:59:59"
$commits = @(git rev-list --all --since=$start --until=$end)

Write-Output "Repository: $repoRoot"
Write-Output "Date: $Date"
Write-Output "Current branch: $(git branch --show-current)"
Write-Output "Commit count across all refs: $($commits.Count)"
Write-Output ''

if ($commits.Count -gt 0) {
    Write-Output '=== Commit graph ==='
    git log --all --since=$start --until=$end --decorate --graph --date=format-local:'%H:%M' --pretty=format:'%ad %h %d %s'
    Write-Output ''
    Write-Output '=== Commit details ==='
    foreach ($commit in $commits) {
        git show --no-renames --format='---%ncommit: %H%ntime: %ad%nsubject: %s%nbody: %b' --date=iso-local --stat --summary $commit
        Write-Output 'branches:'
        git branch --all --contains $commit
    }
}
else {
    Write-Output 'No commits found for the requested date.'
}

if ($IncludeWorkingTree) {
    Write-Output ''
    Write-Output '=== Current uncommitted work ==='
    git status --short
    git diff --stat
}
