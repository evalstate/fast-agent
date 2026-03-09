from __future__ import annotations

import argparse

from fast_agent.core.fastagent import FastAgent


def test_is_acp_server_mode_requires_server_flag_and_acp_transport() -> None:
    agent = FastAgent("TestAgent", parse_cli_args=False)

    agent.args = argparse.Namespace(server=True, transport="acp")
    assert agent._is_acp_server_mode() is True

    agent.args = argparse.Namespace(server=False, transport="acp")
    assert agent._is_acp_server_mode() is False

    agent.args = argparse.Namespace(server=True, transport="stdio")
    assert agent._is_acp_server_mode() is False
