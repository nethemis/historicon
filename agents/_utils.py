"""Small shared helpers for agents/."""

from pathlib import Path


def load_instructions(agent_name: str) -> str:
    """Load an agent's system prompt from `instructions/{agent_name}.txt`.

    Falls back to a generic prompt with a warning if the file is missing.
    """
    instruction_file = (
        Path(__file__).parent.parent / "instructions" / f"{agent_name}.txt"
    )
    if instruction_file.exists():
        return instruction_file.read_text(encoding="utf-8")
    print(f"Instruction file not found: {instruction_file}")
    return f"You are the {agent_name} agent for the HistoriCon Greek podcast."
