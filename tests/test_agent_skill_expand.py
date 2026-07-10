"""Regression test for finding #5: when the model's FINAL answer is solely a
``/<skill>`` line, the auto-expanded turn's returned result must NOT be prefixed
with the raw slash command.

Before the fix, full_response accumulated the raw "/myskill ..." line (line
``full_response += text``) BEFORE _maybe_expand_skill_reply ran, so a headless
JSON result began with the slash-command line prepended to the expanded answer.
"""

import io

import pytest
from rich.console import Console

from spark_code.agent import Agent
from spark_code.context import Context
from spark_code.permissions import PermissionManager
from spark_code.skills.base import Skill, SkillRegistry
from spark_code.tools.base import ToolRegistry


class _SeqModel:
    """Yields a predefined chunk list per chat() call (one per agent round)."""

    total_input_tokens = 0
    total_output_tokens = 0

    def __init__(self, responses):
        self.responses = list(responses)
        self._call = 0

    async def chat(self, **kwargs):
        chunks = (self.responses[self._call]
                  if self._call < len(self.responses) else [])
        self._call += 1
        for c in chunks:
            yield c

    async def close(self):
        pass


def _agent_with_skill(model):
    skills = SkillRegistry()
    skills.register(Skill(
        name="myskill",
        description="a test skill",
        prompt="EXPANDED SKILL BODY: do the analysis.",
    ))
    return Agent(
        model=model,
        context=Context(),
        tools=ToolRegistry(),
        permissions=PermissionManager(mode="trust"),
        console=Console(file=io.StringIO(), force_terminal=True),
        skills=skills,
    )


@pytest.mark.asyncio
async def test_skill_only_reply_returns_expanded_not_slash_line():
    model = _SeqModel([
        # Round 1: the whole answer is just the slash command.
        [{"type": "text", "content": "/myskill focus on X"},
         {"type": "done", "usage": {}}],
        # Round 2: after the expansion is fed back, the real answer.
        [{"type": "text", "content": "Here is the real analysis."},
         {"type": "done", "usage": {}}],
    ])
    agent = _agent_with_skill(model)
    result = await agent.run("please do the thing")

    # The returned result (headless JSON body) is ONLY the expanded answer —
    # the raw "/myskill ..." control line is stripped off.
    assert result == "Here is the real analysis."
    assert "/myskill" not in result

    # Transcript/history stays sane: the slash line was recorded as the
    # assistant turn, and the expansion was fed back as a user message.
    contents = [(m["role"], m.get("content")) for m in agent.context.messages]
    assert ("assistant", "/myskill focus on X") in contents
    assert any(role == "user" and c and c.startswith("EXPANDED SKILL BODY")
               for role, c in contents)


@pytest.mark.asyncio
async def test_multiline_reply_containing_slash_line_returned_verbatim():
    # Guard: a real answer that merely CONTAINS a slash line (multi-line) is not
    # a skill directive — returned verbatim, and the fix must not touch it.
    model = _SeqModel([
        [{"type": "text", "content": "Sure, run /myskill later.\nDone."},
         {"type": "done", "usage": {}}],
    ])
    agent = _agent_with_skill(model)
    result = await agent.run("q")
    assert result == "Sure, run /myskill later.\nDone."
