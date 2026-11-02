import base64
import difflib
import hashlib
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlparse

import requests

from ..observability import log_success
from .config import Settings
from .github_api import GitHubClient
from .repositories import normalize_repo


@dataclass(frozen=True)
class ChangePreview:
    repo: str
    file_path: str
    base_branch: str
    base_sha: str | None
    proposed_sha: str
    diff: str


def validate_file_path(file_path: str) -> PurePosixPath:
    if not file_path or "\0" in file_path or "\\" in file_path:
        raise ValueError(
            "Use a non-empty repository-relative path with forward slashes."
        )

    path = PurePosixPath(file_path)
    if path.is_absolute() or any(
        part in {"", ".", ".."} or part.lower() == ".git"
        for part in file_path.split("/")
    ):
        raise ValueError(
            "The target path must stay inside the repository and cannot target .git."
        )
    return path


def build_change_diff(file_path: str, original: str, proposed: str) -> str:
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
    )
    if not diff:
        raise ValueError("The proposed content does not change the file.")
    return diff


def validate_branch_name(branch_name: str) -> str:
    if not branch_name or branch_name.startswith("-"):
        raise ValueError("The base branch name is invalid.")
    try:
        result = subprocess.run(
            ["git", "check-ref-format", "--branch", branch_name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git must be installed and available on PATH.") from error
    if result.returncode:
        raise ValueError(
            "The base branch returned by GitHub is not a valid Git branch name."
        )
    return branch_name


def preview_file_change(repo: str, file_path: str, new_content: str) -> ChangePreview:
    normalized_repo = normalize_repo(repo)
    if not normalized_repo:
        raise ValueError("Repository must be a valid owner/repo.")
    relative_path = validate_file_path(file_path)
    client = GitHubClient.from_env()
    api_url = f"https://api.github.com/repos/{normalized_repo}"
    repository_data = client._get(api_url).json()
    default_branch = repository_data.get("default_branch")
    if not default_branch:
        raise RuntimeError(
            f"GitHub did not report a default branch for {normalized_repo}."
        )
    validate_branch_name(default_branch)
    contents_url = f"{api_url}/contents/{quote(relative_path.as_posix(), safe='/')}"
    response = client.session.get(
        contents_url,
        params={"ref": default_branch},
        timeout=client.timeout,
    )

    original = ""
    base_sha = None
    if response.status_code == 404:
        response.close()
    else:
        try:
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(
                f"Could not fetch the file preview from GitHub: {error}"
            ) from error
        file_data = response.json()
        if not isinstance(file_data, dict) or file_data.get("type") != "file":
            raise ValueError(
                "The selected path is not a regular file in the base branch."
            )
        if file_data.get("encoding") != "base64" or not isinstance(
            file_data.get("content"), str
        ):
            raise ValueError(
                "GitHub did not return a supported text file representation."
            )
        try:
            encoded_content = "".join(file_data["content"].split())
            original = base64.b64decode(encoded_content, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("The existing file is not valid UTF-8 text.") from error
        base_sha = file_data.get("sha")
        if not isinstance(base_sha, str) or not base_sha:
            raise RuntimeError("GitHub did not return the existing file's revision.")

    diff = build_change_diff(relative_path.as_posix(), original, new_content)
    return ChangePreview(
        repo=normalized_repo,
        file_path=relative_path.as_posix(),
        base_branch=default_branch,
        base_sha=base_sha,
        proposed_sha=hashlib.sha256(new_content.encode("utf-8")).hexdigest(),
        diff=diff,
    )


def _git_blob_sha(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def _branch_name(repo: str, file_path: str, content: str) -> str:
    digest = hashlib.sha256(f"{repo}\0{file_path}\0{content}".encode()).hexdigest()[:16]
    return f"merge-wars/ai-fix-{digest}"


def _validate_pull_url(value: Any, repo: str) -> str:
    parsed_url = urlparse(value) if isinstance(value, str) else None
    url_parts = parsed_url.path.strip("/").split("/") if parsed_url else []
    repo_owner, repo_name = repo.split("/", maxsplit=1)
    if (
        not parsed_url
        or parsed_url.scheme != "https"
        or parsed_url.netloc.lower() != "github.com"
        or len(url_parts) != 4
        or [part.lower() for part in url_parts[:2]] != [repo_owner, repo_name]
        or url_parts[2] != "pull"
        or not url_parts[3].isdigit()
    ):
        raise RuntimeError("GitHub did not return a valid pull request URL.")
    return value


def _existing_pull_request(
    client: GitHubClient,
    repo: str,
    branch_name: str,
    base_branch: str,
) -> str | None:
    owner = repo.split("/", maxsplit=1)[0]
    response = client._get(
        f"https://api.github.com/repos/{repo}/pulls",
        params={"state": "all", "head": f"{owner}:{branch_name}", "base": base_branch},
    )
    existing = response.json()
    if not existing:
        return None
    pull_request = existing[0]
    pull_url = _validate_pull_url(pull_request.get("html_url"), repo)
    if pull_request.get("state") == "open":
        return pull_url
    raise RuntimeError(
        f"A closed pull request already exists for {branch_name}: {pull_url}. "
        "Change the proposed content to create a new branch."
    )


def run_test_command(
    repository_root: Path,
    command: str,
    timeout_seconds: int = 300,
) -> str:
    if not command.strip():
        raise RuntimeError(
            "PR_TEST_COMMAND is required. Set it to the repository's test command before creating a PR."
        )
    arguments = shlex.split(command, posix=os.name != "nt")
    if not arguments:
        raise ValueError("PR_TEST_COMMAND must contain an executable.")
    test_environment = os.environ.copy()
    for name in tuple(test_environment):
        upper_name = name.upper()
        if any(
            secret in upper_name
            for secret in ("TOKEN", "API_KEY", "SECRET", "PASSWORD")
        ) or upper_name.startswith("GIT_CONFIG_"):
            test_environment.pop(name)
    try:
        result = subprocess.run(
            arguments,
            cwd=repository_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=test_environment,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Could not start PR_TEST_COMMAND: {arguments[0]} was not found."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"PR_TEST_COMMAND exceeded its {timeout_seconds}-second timeout."
        ) from error
    if result.returncode:
        output = "\n".join(
            part for part in (result.stdout, result.stderr) if part
        ).strip()
        raise RuntimeError(
            f"PR_TEST_COMMAND failed with exit code {result.returncode}:\n{output[-6000:]}"
        )
    return result.stdout.strip() or "Test command completed successfully."


def create_pull_request(
    repo: str,
    file_path: str,
    new_content: str,
    title: str,
    body: str,
    preview: ChangePreview | None = None,
    test_command: str | None = None,
) -> str:
    normalized_repo = normalize_repo(repo)
    if not normalized_repo:
        raise ValueError("Repository must be a valid owner/repo.")
    relative_path = validate_file_path(file_path)
    settings = Settings.from_env()
    client = GitHubClient.from_env()
    api_url = f"https://api.github.com/repos/{normalized_repo}"
    repository_data = client._get(api_url).json()
    default_branch = repository_data.get("default_branch")
    if not default_branch:
        raise RuntimeError(
            f"GitHub did not report a default branch for {normalized_repo}."
        )
    validate_branch_name(default_branch)
    if preview and (
        preview.repo != normalized_repo
        or preview.file_path != relative_path.as_posix()
        or preview.base_branch != default_branch
        or preview.proposed_sha
        != hashlib.sha256(new_content.encode("utf-8")).hexdigest()
    ):
        raise ValueError(
            "The repository, file, content, or default branch changed after the diff preview."
        )

    existing_branch_name = _branch_name(
        normalized_repo,
        relative_path.as_posix(),
        new_content,
    )
    existing_pull = _existing_pull_request(
        client,
        normalized_repo,
        existing_branch_name,
        default_branch,
    )
    if existing_pull:
        return existing_pull

    auth_value = (
        "basic "
        + base64.b64encode(f"x-access-token:{settings.github_token}".encode()).decode()
    )
    git_env = os.environ.copy()
    git_env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: {auth_value}",
        }
    )
    branch_name = existing_branch_name
    clone_url = f"https://github.com/{normalized_repo}.git"

    import git

    with tempfile.TemporaryDirectory() as temporary_directory:
        repository = git.Repo.clone_from(clone_url, temporary_directory, env=git_env)
        remote_branch = repository.git.ls_remote(
            "--heads",
            "origin",
            f"refs/heads/{branch_name}",
        )
        if remote_branch:
            raise RuntimeError(
                f"Branch {branch_name} already exists without an open matching PR. "
                "Review or remove that branch before retrying."
            )
        repository.git.checkout("-b", branch_name, f"origin/{default_branch}")
        repository.git.config("user.name", "Merge Wars AI")
        repository.git.config("user.email", "merge-wars-ai@users.noreply.github.com")

        root = Path(temporary_directory).resolve()
        destination_candidate = root
        for part in relative_path.parts:
            destination_candidate = destination_candidate / part
            if destination_candidate.is_symlink():
                raise ValueError("The target path cannot traverse repository symlinks.")
        destination = destination_candidate.resolve()
        if os.path.commonpath((str(root), str(destination))) != str(root):
            raise ValueError("The target path resolves outside the repository.")
        if preview:
            if preview.base_sha is None:
                if destination.exists():
                    raise RuntimeError(
                        "The file was added after the diff preview; preview it again."
                    )
            elif (
                not destination.is_file()
                or _git_blob_sha(destination.read_bytes()) != preview.base_sha
            ):
                raise RuntimeError(
                    "The file changed after the diff preview; preview it again."
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(new_content, encoding="utf-8")

        run_test_command(
            root,
            test_command or settings.pr_test_command,
            settings.pr_test_timeout_seconds,
        )
        repository.git.add("--", relative_path.as_posix())
        if not repository.is_dirty(untracked_files=True):
            raise ValueError("The proposed content does not change the repository.")
        repository.index.commit(title)
        repository.git.push("--set-upstream", "origin", branch_name)

    pull_request_response = client.session.post(
        f"{api_url}/pulls",
        json={
            "title": title,
            "head": branch_name,
            "base": default_branch,
            "body": f"{body}\n\nChecks: `{test_command or settings.pr_test_command}` passed.",
        },
        timeout=client.timeout,
    )
    if pull_request_response.status_code == 422:
        existing_pull = _existing_pull_request(
            client,
            normalized_repo,
            branch_name,
            default_branch,
        )
        if existing_pull:
            return existing_pull
    try:
        pull_request_response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"GitHub could not create the pull request: {error}"
        ) from error
    pull_request: Any = pull_request_response.json()
    pull_request_url = _validate_pull_url(pull_request.get("html_url"), normalized_repo)
    log_success(f"Pull request created for {normalized_repo}")
    return pull_request_url
