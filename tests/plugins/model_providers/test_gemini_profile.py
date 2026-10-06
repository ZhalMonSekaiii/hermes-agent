"""Contract tests for the native Google Gemini provider profile."""

from __future__ import annotations

import pytest


@pytest.fixture
def gemini_profile():
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("gemini")
    assert profile is not None, "gemini provider profile must be registered"
    return profile


def test_native_gemini_auxiliary_default_is_in_curated_catalog(gemini_profile):
    """The profile's default_aux_model must stay in lockstep with the curated
    model picker catalog — whatever model the default points at has to be
    one the picker can actually offer. Deliberately durable against future
    model-generation bumps: it does not pin either side to a frozen
    model-name string, only to the invariant that they never drift apart.
    """
    from hermes_cli.models import _PROVIDER_MODELS

    assert gemini_profile.default_aux_model in _PROVIDER_MODELS["gemini"]


def test_canonical_gemini_is_byte_role_stable_for_system_instructions(gemini_profile):
    """Canonical Gemini must preserve system instructions natively and byte-for-byte.

    Issue #76783 is an opt-in relay compatibility workaround for proxies fronting Gemini.
    Canonical Gemini's native API has first-class system_instruction authority, so
    prepare_messages must never mangle system prompts or embed instructions in user turns.
    """
    assert gemini_profile.system_prompt_mode == "system"

    long_system_text = "You are Hermes Agent. System prompt rules: " + ("abcde12345 " * 50)
    assert len(long_system_text) > 500

    messages = [
        {"role": "system", "content": long_system_text},
        {"role": "user", "content": "Hello, world!"},
    ]

    prepared = gemini_profile.prepare_messages(messages)
    # Byte/role stability
    assert prepared == messages
    assert prepared[0]["role"] == "system"
    assert prepared[0]["content"] == long_system_text
    assert prepared[1]["role"] == "user"
    assert prepared[1]["content"] == "Hello, world!"
    assert "<hermes_instructions>" not in str(prepared)


def test_canonical_gemini_wire_kwargs_preserve_system_role(gemini_profile):
    """The ChatCompletions transport preserves canonical Gemini's system instructions."""
    from agent.transports.chat_completions import ChatCompletionsTransport

    long_system_text = "You are Hermes Agent. Native system instruction. " * 20
    assert len(long_system_text) > 400

    messages = [
        {"role": "system", "content": long_system_text},
        {"role": "user", "content": "Ping"},
    ]

    transport = ChatCompletionsTransport()
    kwargs = transport.build_kwargs(
        model="gemini-2.5-flash",
        messages=messages,
        provider_profile=gemini_profile,
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )

    wire_messages = kwargs["messages"]
    # System instruction must be in system role and not demoted into user turn
    assert wire_messages[0]["role"] == "system"
    assert wire_messages[0]["content"] == long_system_text
    assert wire_messages[1]["role"] == "user"
    assert wire_messages[1]["content"] == "Ping"
    assert "<hermes_instructions>" not in wire_messages[1]["content"]

