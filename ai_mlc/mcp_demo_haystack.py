import os

from dotenv import load_dotenv

from ai_mlc.github_agent.agent import agent_fetch, setup_mcp_agent

load_dotenv()


def main() -> None:
    agent = setup_mcp_agent()
    repo = os.getenv("DEMO_REPO", "octocat/Hello-World")
    answer = agent_fetch(
        agent,
        f"Use list_issues for {repo}. Summarize the latest open issues.",
    )
    print(answer)


if __name__ == "__main__":
    main()
