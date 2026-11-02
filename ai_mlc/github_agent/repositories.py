import re


_REPO_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?/[A-Za-z0-9._-]+$")
_URL_PATTERN = re.compile(
    r"https?://github\.com/([A-Za-z0-9-]+)/([A-Za-z0-9._-]+)(?:/)?",
    re.IGNORECASE,
)
_REPO_IN_TEXT_PATTERN = re.compile(r"(?<![\w.-])([A-Za-z0-9-]+/[A-Za-z0-9._-]+)")


def normalize_repo(text: str | None) -> str | None:
    if not text:
        return None

    candidate = text.strip()
    url_match = _URL_PATTERN.fullmatch(candidate)
    if url_match:
        owner, repository = url_match.groups()
        if repository.endswith(".git"):
            repository = repository[:-4]
        candidate = f"{owner}/{repository}"

    if not _REPO_PATTERN.fullmatch(candidate):
        return None
    owner, repository = candidate.split("/", maxsplit=1)
    if repository in {".", ".."}:
        return None
    return f"{owner.lower()}/{repository.lower()}"


def normalize_repos(multi_text: str | None) -> list[str]:
    if not multi_text:
        return []

    repos: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[\n,]", multi_text):
        repo = normalize_repo(part)
        if repo and repo not in seen:
            repos.append(repo)
            seen.add(repo)
    return repos


def extract_repos(text: str) -> list[str]:
    repos: list[str] = []
    seen: set[str] = set()
    for match in _URL_PATTERN.finditer(text):
        repo = normalize_repo(match.group(0))
        if repo and repo not in seen:
            repos.append(repo)
            seen.add(repo)
    remaining_text = _URL_PATTERN.sub(" ", text)
    for match in _REPO_IN_TEXT_PATTERN.finditer(remaining_text):
        repo = normalize_repo(match.group(1))
        if repo and repo not in seen:
            repos.append(repo)
            seen.add(repo)
    return repos


def infer_intent(query: str) -> str:
    text = query.lower()
    if any(word in text for word in ("error", "bug", "fix")):
        return "find_errors"
    if "how many" in text or "count" in text:
        return "counts"
    if "pull" in text or re.search(r"\bprs?\b", text):
        return "prs"
    if "commit" in text:
        return "commits"
    if "discussion" in text:
        return "discussions"
    if "issue" in text:
        return "issues"
    if any(word in text for word in ("summary", "summarize", "health", "report")):
        return "summary"
    return "general"
