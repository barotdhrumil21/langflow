"""Native model output/legacy selection paths preserve synchronous compatibility."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from lfx.components.models_and_agents import agent as agent_module
from lfx.components.models_and_agents.agent import AgentComponent
from lfx.utils.async_helpers import async_call_method

SELECTION = [{"name": "gpt-4o-mini", "provider": "OpenAI", "metadata": {}}]


def policy():
    return SimpleNamespace(
        require=Mock(), allows=lambda name: name == "OpenAI", allows_model=lambda *_args, **_kwargs: True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("override_kind", ["subclass", "instance"])
async def test_agent_sync_model_overrides_preserve_selection_endpoint_and_context(monkeypatch, override_kind):
    import threading
    from contextvars import ContextVar
    from types import MethodType

    context = ContextVar("agent-extension-context", default="missing")
    caller_thread = threading.get_ident()
    selected = [{"name": "owned-model", "provider": "OpenAI", "metadata": {}}]
    chosen = SimpleNamespace(model_name="owned-model", base_url="https://owned.example/v1")
    observed = []

    def custom_selection(self):
        assert self.user_id == "runtime-owner"
        assert context.get() == "owner"
        return deepcopy(selected)

    def custom_model(self):
        assert context.get() == "owner"
        observed.append((deepcopy(self.model), threading.get_ident()))
        return chosen

    class ExtendedAgent(AgentComponent):
        _resolve_selected_model = custom_selection
        _get_llm = custom_model

    cls = ExtendedAgent if override_kind == "subclass" else AgentComponent
    c = cls(_user_id="runtime-owner", model=SELECTION)
    if override_kind == "instance":
        monkeypatch.setattr(c, "_resolve_selected_model", MethodType(custom_selection, c))
        monkeypatch.setattr(c, "_get_llm", MethodType(custom_model, c))
    c.set_attributes({"tools": [], "add_current_date_tool": False, "chat_history": []})
    monkeypatch.setattr(
        "lfx.services.model_provider_policy.aresolve_model_provider_policy", AsyncMock(return_value=policy())
    )
    native = AsyncMock(side_effect=AssertionError("custom synchronous model override bypassed"))
    monkeypatch.setattr(agent_module, "aget_llm", native)
    monkeypatch.setattr(type(c), "get_memory_data", AsyncMock(return_value=[]))
    token = context.set("owner")
    try:
        llm, _history, _tools = await c.get_agent_requirements()
        provider, name, _connected = await async_call_method(c, "_selected_model_remediation_context")
    finally:
        context.reset(token)
    assert llm is chosen
    assert llm.base_url == "https://owned.example/v1"
    assert provider == "OpenAI"
    assert name == "owned-model"
    assert observed[0][0] == selected
    assert observed[0][1] != caller_thread
    native.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("override_kind", ["subclass", "instance"])
async def test_decorated_agent_sync_overrides_are_not_skipped(monkeypatch, override_kind):
    from contextvars import ContextVar
    from functools import wraps
    from types import MethodType

    context = ContextVar("decorated-owner", default="missing")
    selected = [{"name": "decorated-model", "provider": "OpenAI", "metadata": {}}]
    chosen = SimpleNamespace(model_name="decorated-model", base_url="https://decorated.example/v1")
    calls = []

    @wraps(AgentComponent._resolve_selected_model)
    def select(self):
        assert self.user_id == "runtime-owner"
        assert context.get() == "owner"
        calls.append("select")
        return deepcopy(selected)

    @wraps(AgentComponent._get_llm)
    def build(self):
        assert self.model == selected
        assert context.get() == "owner"
        calls.append("build")
        return chosen

    class DecoratedAgent(AgentComponent):
        _resolve_selected_model = select
        _get_llm = build

    c = (DecoratedAgent if override_kind == "subclass" else AgentComponent)(_user_id="runtime-owner", model=SELECTION)
    if override_kind == "instance":
        monkeypatch.setattr(c, "_resolve_selected_model", MethodType(select, c))
        monkeypatch.setattr(c, "_get_llm", MethodType(build, c))
    c.set_attributes({"tools": [], "add_current_date_tool": False, "chat_history": []})
    monkeypatch.setattr(
        "lfx.services.model_provider_policy.aresolve_model_provider_policy", AsyncMock(return_value=policy())
    )
    native = AsyncMock(side_effect=AssertionError("decorated override skipped"))
    monkeypatch.setattr(agent_module, "aget_llm", native)
    monkeypatch.setattr(type(c), "get_memory_data", AsyncMock(return_value=[]))
    token = context.set("owner")
    try:
        llm, _history, _tools = await c.get_agent_requirements()
    finally:
        context.reset(token)
    assert llm is chosen
    assert calls == ["select", "build"]
    native.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("override_kind", ["subclass", "instance", "decorated"])
async def test_custom_remediation_context_keeps_provenance_and_connected_target(monkeypatch, override_kind):
    from contextvars import ContextVar
    from functools import wraps
    from types import MethodType

    from lfx.schema.message import Message

    context = ContextVar("remediation-owner", default="missing")
    target = SimpleNamespace(model_name="owned-connected-model")

    def provenance(self):
        assert self.user_id == "runtime-owner"
        assert context.get() == "owner"
        return "OpenAI", "owned-connected-model", target

    if override_kind == "decorated":
        provenance = wraps(AgentComponent._selected_model_remediation_context)(provenance)

    class CustomContextAgent(AgentComponent):
        _selected_model_remediation_context = provenance

    c = (AgentComponent if override_kind == "instance" else CustomContextAgent)(
        _user_id="runtime-owner", model=SELECTION
    )
    if override_kind == "instance":
        monkeypatch.setattr(c, "_selected_model_remediation_context", MethodType(provenance, c))
    monkeypatch.setattr(
        "lfx.services.model_provider_policy.aresolve_model_provider_policy", AsyncMock(return_value=policy())
    )
    remediation = SimpleNamespace(name="test-remediation", overrides={"temperature": 0})
    find = Mock(side_effect=[remediation])
    apply = Mock(return_value=True)
    monkeypatch.setattr("lfx.base.models.model_remediation.find_remediation", find)
    monkeypatch.setattr("lfx.base.models.model_remediation.apply_overrides_to_model", apply)
    answer = Message(text="answer")
    run = AsyncMock(side_effect=[RuntimeError("request validation"), answer])
    token = context.set("owner")
    try:
        assert await c._run_agent_with_model_remediation(run) is answer
    finally:
        context.reset(token)
    apply.assert_called_once_with(target, remediation.overrides)
    assert find.call_args.args[1] == "OpenAI"
    assert run.await_count == 2


@pytest.mark.asyncio
async def test_legacy_agent_selection_and_credentials_use_native_async_hooks(monkeypatch):
    c = AgentComponent(_user_id="runtime-owner", model=[], agent_llm="OpenAI", model_name="gpt-4o-mini")
    c.set_attributes({"tools": [], "add_current_date_tool": False, "chat_history": []})
    p = policy()
    monkeypatch.setattr("lfx.services.model_provider_policy.aresolve_model_provider_policy", AsyncMock(return_value=p))
    options = AsyncMock(return_value=SELECTION)
    factory = AsyncMock(return_value=object())
    monkeypatch.setattr(agent_module, "aget_language_model_options", options)
    monkeypatch.setattr(agent_module, "aget_llm", factory)
    monkeypatch.setattr(
        agent_module, "get_language_model_options", Mock(side_effect=AssertionError("sync legacy catalog forbidden"))
    )
    monkeypatch.setattr(agent_module, "get_llm", Mock(side_effect=AssertionError("sync model factory forbidden")))
    monkeypatch.setattr(type(c), "get_memory_data", AsyncMock(return_value=[]))
    llm, history, _tools = await c.get_agent_requirements()
    assert llm is factory.return_value
    assert history == []
    options.assert_awaited_once()
    factory.assert_awaited_once()


@pytest.mark.asyncio
async def test_legacy_direct_agent_denial_precedes_catalog_and_model(monkeypatch):
    c = AgentComponent(_user_id="runtime-owner", model=[], agent_llm="OpenAI", model_name="gpt-4o-mini")
    p = policy()
    p.require.side_effect = PermissionError("denied")
    monkeypatch.setattr("lfx.services.model_provider_policy.aresolve_model_provider_policy", AsyncMock(return_value=p))
    options = AsyncMock(side_effect=AssertionError("legacy catalog before denial"))
    factory = AsyncMock()
    monkeypatch.setattr(agent_module, "aget_language_model_options", options)
    monkeypatch.setattr(agent_module, "aget_llm", factory)
    with pytest.raises(PermissionError):
        await c.message_response()
    options.assert_not_awaited()
    factory.assert_not_awaited()
