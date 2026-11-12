"""Attachment contracts using synthetic credentials and an in-memory HTTP transport."""

import asyncio
import base64
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from PIL import Image

from fast_agent.llm.provider.copilot import images
from fast_agent.llm.provider.copilot.broker import CopilotEndpoint


@pytest.fixture
def endpoint():
    return CopilotEndpoint(
        model_id="test",
        wire_api="messages",
        transport="sse",
        base_url="https://unused.example",
        headers={"Authorization": "Bearer synthetic"},
    )


def image_data(size=(20, 10), color=(0, 0, 0)):
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def block(data):
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(data).decode(),
        },
        "extra": "preserved",
    }


def mock_http(monkeypatch, handler):
    original = httpx.AsyncClient

    def factory(**kwargs):
        assert kwargs == {"timeout": 30, "trust_env": False, "follow_redirects": False}
        return original(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(images.httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_reuse_nested_content_and_immutable_history(monkeypatch, endpoint):
    data = image_data()
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "POST"
        assert str(request.url).startswith("https://uploads.github.com/copilot/chat/attachments?")
        assert request.url.params["content_type"] == "image/png"
        assert request.headers["authorization"] == "Bearer synthetic"
        assert request.headers["content-type"] == "application/octet-stream"
        assert request.content == data
        return httpx.Response(
            201, json={"url": "https://attachment.example/image?signature=secret"}
        )

    mock_http(monkeypatch, handler)
    image = block(data)
    payload: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    image,
                    {
                        "type": "tool_result",
                        "content": [image],
                        "output": [image],
                    },
                    {"type": "tool_use", "input": {"content": [image]}, "content": [image]},
                ],
            }
        ],
        "metadata": {"content": [image]},
    }
    before = deepcopy(payload)
    uploads = images.CopilotImageUploads()
    result = await uploads.normalize(payload, endpoint)
    assert payload == before
    content = result["messages"][0]["content"]
    assert content[0]["source"]["type"] == "url"
    assert content[0]["extra"] == "preserved"
    assert content[1]["content"][0] == content[0]
    assert content[1]["output"][0] == content[0]
    assert content[2] == before["messages"][0]["content"][2]
    assert result["metadata"] == before["metadata"]
    await uploads.normalize(payload, endpoint)
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_responses_and_credential_scoping(monkeypatch, endpoint):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(201, json={"url": "https://attachment.example/image"})

    mock_http(monkeypatch, handler)
    endpoint = replace(endpoint, wire_api="responses")
    image = {
        "type": "input_image",
        "detail": "high",
        "image_url": "data:image/png;base64," + base64.b64encode(image_data()).decode(),
    }
    payload: dict[str, Any] = {
        "input": [
            {"role": "user", "content": [image]},
            {"type": "function_call_output", "output": [image]},
        ]
    }
    uploads = images.CopilotImageUploads()
    result = await uploads.normalize(payload, endpoint)
    assert result["input"][0]["content"][0] == {
        **image,
        "image_url": "https://attachment.example/image",
    }
    assert result["input"][1]["output"][0]["image_url"].startswith("https:")
    await uploads.normalize(payload, replace(endpoint, headers={"Authorization": "Bearer other"}))
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "network", "url", "json"])
async def test_resizing_and_safe_inline_fallback(monkeypatch, endpoint, caplog, failure):
    requests = []

    def handler(request):
        requests.append(request)
        with Image.open(BytesIO(request.content)) as image:
            assert image.size == (2000, 1000)
        if failure == "network":
            raise httpx.ConnectError("secret signed URL", request=request)
        if failure == "http":
            return httpx.Response(403, text="secret")
        if failure == "json":
            return httpx.Response(201, text="secret")
        return httpx.Response(201, json={"url": "http://unsafe.example/secret"})

    mock_http(monkeypatch, handler)
    payload: dict[str, Any] = {
        "messages": [
            {
                "content": [
                    block(image_data((2400, 1200))),
                    {"type": "tool_result", "content": [block(image_data((3000, 1500)))]},
                ]
            }
        ]
    }
    before = deepcopy(payload)
    result = await images.CopilotImageUploads().normalize(payload, endpoint)
    source = result["messages"][0]["content"][0]["source"]
    assert source["type"] == "base64"
    with Image.open(BytesIO(base64.b64decode(source["data"]))) as image:
        assert image.size == (2000, 1000)
    nested_source = result["messages"][0]["content"][1]["content"][0]["source"]
    assert nested_source["type"] == "base64"
    with Image.open(BytesIO(base64.b64decode(nested_source["data"]))) as image:
        assert image.size == (2000, 1000)
    assert len(requests) == 1
    assert payload == before
    assert "retaining inline" in caplog.text
    assert "secret" not in caplog.text


@pytest.mark.asyncio
async def test_progress_reports_only_uncached_uploads(monkeypatch, endpoint):
    mock_http(
        monkeypatch,
        lambda request: httpx.Response(201, json={"url": "https://attachment.example/image"}),
    )
    first, second = block(image_data()), block(image_data(color=(1, 0, 0)))
    uploads = images.CopilotImageUploads()
    progress: list[tuple[int, int]] = []

    async def send(*content):
        await uploads.normalize(
            {"messages": [{"content": list(content)}]},
            endpoint,
            on_progress=lambda count, total: progress.append((count, total)),
        )

    # A repeated history image uploads once.
    await send(first, first, second)
    assert progress == [(1, 2), (2, 2)]
    progress.clear()
    await send(first, second)
    assert progress == []
    third = block(image_data(color=(2, 0, 0)))
    await send(first, second, third)
    assert progress == [(1, 1)]


@pytest.mark.asyncio
async def test_cache_expiry_and_bound(monkeypatch, endpoint):
    count = 0
    clock = 0.0

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(201, json={"url": "https://attachment.example/image"})

    mock_http(monkeypatch, handler)
    monkeypatch.setattr(images.time, "monotonic", lambda: clock)
    uploads = images.CopilotImageUploads()
    endpoint = replace(endpoint, wire_api="responses")

    async def send(index):
        await uploads.normalize({"messages": [{"content": [block(str(index).encode())]}]}, endpoint)

    await send(0)
    await send(0)
    assert count == 1
    clock = 1801
    await send(0)
    assert count == 2
    for index in range(1, 129):
        await send(index)
    await send(0)
    assert count == 131


@pytest.mark.asyncio
async def test_cancellation_propagates(monkeypatch, endpoint):
    def handler(request):
        raise asyncio.CancelledError

    mock_http(monkeypatch, handler)
    with pytest.raises(asyncio.CancelledError):
        await images.CopilotImageUploads().normalize(
            {"messages": [{"content": [block(image_data())]}]}, endpoint
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["messages", "responses"])
async def test_request_url_budget_nested_mixed_images(monkeypatch, endpoint, route):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(201, json={"url": "https://attachment.example/image"})

    mock_http(monkeypatch, handler)
    endpoint = replace(endpoint, wire_api=route)
    data = image_data((2400, 1200))
    inline = block(data)
    response_inline = {
        "type": "input_image",
        "image_url": "data:image/png;base64," + base64.b64encode(data).decode(),
    }
    existing = {"type": "image", "source": {"type": "url", "url": "https://existing/image"}}
    response_url = {"type": "input_image", "image_url": "https://existing/other"}
    # Existing URLs come last, so they must be counted before the first upload.
    payload = {
        "messages": [
            {
                "content": [
                    *[inline, response_inline] * 11,
                    {"type": "tool_result", "content": [existing], "output": [response_url]},
                    {"type": "tool_use", "input": {"content": [existing] * 30}},
                ]
            }
        ]
    }
    before = deepcopy(payload)
    uploads = images.CopilotImageUploads()
    resize = uploads._resize
    resize_calls = []

    def tracked_resize(data, mime):
        resize_calls.append(mime)
        return resize(data, mime)

    monkeypatch.setattr(uploads, "_resize", tracked_resize)
    for _ in range(2):
        result = await uploads.normalize(payload, endpoint)
        content = result["messages"][0]["content"]
        blocks = content[:22] + content[22]["content"] + content[22]["output"]
        urls = 0
        inlines = 0
        for image_block in blocks:
            if image_block["type"] == "image":
                source = image_block["source"]
                is_url = source["type"] == "url"
                encoded = source.get("data")
            else:
                url = image_block["image_url"]
                is_url = not url.startswith("data:")
                encoded = None if is_url else url.split(",", 1)[1]
            if is_url:
                urls += 1
            else:
                inlines += 1
                assert isinstance(encoded, str)
                with Image.open(BytesIO(base64.b64decode(encoded))) as image:
                    assert image.size == ((2000, 1000) if route == "messages" else (2400, 1200))
        assert urls == 20
        assert inlines == 4
        assert content[23] == before["messages"][0]["content"][23]
        assert payload == before
    assert len(requests) == 1
    assert len(resize_calls) == (2 if route == "messages" else 0)


@pytest.mark.parametrize("format", ["JPEG", "PNG", "GIF"])
def test_resize_format_and_small_image_passthrough(format):
    for size in [(20, 10), (2400, 1200)]:
        stream = BytesIO()
        Image.new("RGB", size).save(stream, format=format)
        data = stream.getvalue()
        mime = "image/" + format.lower()
        resized, result_mime = images.CopilotImageUploads._resize(data, mime)
        if size == (20, 10):
            assert (resized, result_mime) == (data, mime)
        else:
            expected = "JPEG" if format == "JPEG" else "PNG"
            assert result_mime == "image/" + expected.lower()
            with Image.open(BytesIO(resized)) as image:
                assert image.format == expected
                assert image.size == (2000, 1000)


@pytest.mark.asyncio
async def test_responses_default_detail_and_explicit_detail(monkeypatch, endpoint):
    mock_http(monkeypatch, lambda request: httpx.Response(503))
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "https://attachment.example/image"},
                    {
                        "type": "input_image",
                        "image_url": "https://attachment.example/image",
                        "detail": "high",
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,"
                        + base64.b64encode(image_data()).decode(),
                    },
                ],
            }
        ]
    }
    original = deepcopy(payload)
    result = await images.CopilotImageUploads().normalize(
        payload, replace(endpoint, wire_api="responses")
    )
    assert [block["detail"] for block in result["input"][0]["content"]] == ["auto", "high", "auto"]
    assert payload == original


@pytest.mark.asyncio
@pytest.mark.parametrize("in_flight", [False, True])
@pytest.mark.parametrize("route", ["messages", "responses"])
async def test_request_upload_deadline(monkeypatch, endpoint, in_flight, route):
    clock = 100.0
    requests = []
    timeouts = []

    def handler(request):
        nonlocal clock
        requests.append(request)
        clock += 40 if in_flight else 60
        return httpx.Response(201, json={"url": "https://attachment.example/image"})

    @asynccontextmanager
    async def timeout(delay):
        timeouts.append(delay)
        if len(timeouts) == 2:
            raise TimeoutError
        yield

    mock_http(monkeypatch, handler)
    monkeypatch.setattr(images, "time", SimpleNamespace(monotonic=lambda: clock))
    monkeypatch.setattr(images.asyncio, "timeout", timeout)
    payload = {
        "messages": [
            {
                "content": [
                    # Distinct pixels: identical resized bytes would reuse the first URL.
                    block(image_data((2400, 1200))),
                    block(image_data((3000, 1500), (1, 0, 0))),
                    block(image_data((2600, 1300), (2, 0, 0))),
                ]
            }
        ]
    }
    before = deepcopy(payload)
    result = await images.CopilotImageUploads().normalize(
        payload, replace(endpoint, wire_api=route)
    )
    assert len(requests) == 1
    assert timeouts == ([60, 20] if in_flight else [60])
    content = result["messages"][0]["content"]
    assert content[0]["source"]["type"] == "url"
    for image_block, original_size in zip(content[1:], [(3000, 1500), (2600, 1300)], strict=True):
        source = image_block["source"]
        assert source["type"] == "base64"
        with Image.open(BytesIO(base64.b64decode(source["data"]))) as image:
            assert image.size == ((2000, 1000) if route == "messages" else original_size)
    assert payload == before
