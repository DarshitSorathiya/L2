# Merge Wars

From the repository root, install the pinned dependencies and configure credentials:

```powershell
python -m pip install -r ai_mlc\requirements.txt
Copy-Item .env.example .env
```

Set `GITHUB_PERSONAL_ACCESS_TOKEN` and `GOOGLE_API_KEY` in `.env`. For PR creation, also set `PR_TEST_COMMAND` to a test command for the target repository; it must pass before any push.

Run from the repository root:

```powershell
python -m ai_mlc.cli
python -m streamlit run ai_mlc\stream_app.py
python -m ai_mlc.mcp_demo_haystack
```

The shared Haystack agent combines MCP source/repository tools with structured GitHub API tools for metadata, issues, pull requests, labels, review history, filters, pagination, and linked summaries. PR creation shows a default-branch diff and requires explicit approval.

For token permissions, configuration options, test commands, and CI details, see the root [README](../README.md).
