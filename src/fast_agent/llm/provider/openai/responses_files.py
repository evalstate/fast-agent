from __future__ import annotations

import base64
import binascii
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from fast_agent.mcp.mime_utils import guess_mime_type


class ResponsesFileMixin:
    if TYPE_CHECKING:
        _file_id_cache: dict[str, str]

    @staticmethod
    def _split_data_url(data_url: str) -> tuple[str | None, str | None]:
        if not data_url.startswith("data:"):
            return None, None
        header, _, payload = data_url.partition(",")
        if ";base64" not in header or not payload:
            return None, None
        mime_type = header[5:].split(";", 1)[0] or None
        return mime_type, payload

    def _decode_file_data(self, raw_data: str) -> tuple[bytes | None, str | None]:
        mime_type, payload = self._split_data_url(raw_data)
        if payload is None:
            payload = raw_data
        try:
            return base64.b64decode(payload, validate=True), mime_type
        except (binascii.Error, ValueError):
            try:
                return base64.b64decode(payload), mime_type
            except (binascii.Error, ValueError):
                return None, mime_type

    @staticmethod
    def _file_cache_key(data: bytes, filename: str | None, mime_type: str | None) -> str:
        digest = hashlib.sha256(data).hexdigest()
        if filename:
            digest = f"{filename}:{digest}"
        if mime_type:
            digest = f"{mime_type}:{digest}"
        return digest

    async def _upload_file_bytes(
        self,
        client: AsyncOpenAI,
        data: bytes,
        filename: str | None,
        mime_type: str | None,
    ) -> str:
        cache_key = self._file_cache_key(data, filename, mime_type)
        cached = self._file_id_cache.get(cache_key)
        if cached:
            return cached

        if filename and mime_type:
            file_param: Any = (filename, data, mime_type)
        elif filename:
            file_param = (filename, data)
        elif mime_type:
            file_param = ("file", data, mime_type)
        else:
            file_param = data

        file_obj = await client.files.create(file=file_param, purpose="user_data")
        self._file_id_cache[cache_key] = file_obj.id
        return file_obj.id

    @staticmethod
    def _input_image_file_id_part(
        file_id: str,
        detail: Any,
    ) -> dict[str, Any]:
        new_part: dict[str, Any] = {"type": "input_image", "file_id": file_id}
        if detail:
            new_part["detail"] = detail
        return new_part

    async def _normalize_input_image_part(
        self,
        client: AsyncOpenAI,
        part: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        detail = part.get("detail")
        image_url = part.get("image_url")
        file_id = part.get("file_id")
        if file_id:
            return self._input_image_file_id_part(file_id, detail), bool(image_url)
        if not isinstance(image_url, str) or not image_url.startswith("file://"):
            return part, False

        local_path = Path(image_url[len("file://") :])
        if not local_path.exists():
            return part, False

        data_bytes = local_path.read_bytes()
        mime_type = guess_mime_type(local_path.name)
        uploaded_file_id = await self._upload_file_bytes(
            client, data_bytes, local_path.name, mime_type
        )
        return self._input_image_file_id_part(uploaded_file_id, detail), True

    async def _file_id_part_from_data(
        self,
        client: AsyncOpenAI,
        file_data: str,
        filename: str | None,
    ) -> dict[str, Any] | None:
        data_bytes, detected_mime = self._decode_file_data(file_data)
        if data_bytes is None:
            return None

        mime_type = detected_mime or (guess_mime_type(filename) if filename else None)
        file_id = await self._upload_file_bytes(client, data_bytes, filename, mime_type)
        return {"type": "input_file", "file_id": file_id}

    def _file_bytes_from_url(
        self,
        file_url: str,
        filename: str | None,
    ) -> tuple[bytes | None, str | None, str | None]:
        if file_url.startswith("data:"):
            data_bytes, mime_type = self._decode_file_data(file_url)
            return data_bytes, filename, mime_type
        if not file_url.startswith("file://"):
            return None, filename, None

        local_path = Path(file_url[len("file://") :])
        if not local_path.exists():
            return None, filename, None

        resolved_filename = filename or local_path.name
        return local_path.read_bytes(), resolved_filename, guess_mime_type(local_path.name)

    async def _file_id_part_from_url(
        self,
        client: AsyncOpenAI,
        file_url: str,
        filename: str | None,
    ) -> dict[str, Any] | None:
        data_bytes, resolved_filename, mime_type = self._file_bytes_from_url(file_url, filename)
        if data_bytes is None:
            return None

        file_id = await self._upload_file_bytes(client, data_bytes, resolved_filename, mime_type)
        return {"type": "input_file", "file_id": file_id}

    async def _normalize_input_file_part(
        self,
        client: AsyncOpenAI,
        part: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        file_id = part.get("file_id")
        if file_id:
            new_part = {"type": "input_file", "file_id": file_id}
            changed = bool(part.get("filename") or part.get("file_url") or part.get("file_data"))
            return new_part, changed

        filename = part.get("filename")
        file_data = part.get("file_data")
        if file_data:
            new_part = await self._file_id_part_from_data(client, file_data, filename)
            return (new_part, True) if new_part else (part, False)

        file_url = part.get("file_url")
        if file_url:
            new_part = await self._file_id_part_from_url(client, file_url, filename)
            return (new_part, True) if new_part else (part, False)

        return part, False

    async def _normalize_input_part(
        self,
        client: AsyncOpenAI,
        part: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        match part.get("type"):
            case "input_image":
                return await self._normalize_input_image_part(client, part)
            case "input_file":
                return await self._normalize_input_file_part(client, part)
            case _:
                return part, False

    async def _normalize_content_parts(
        self,
        client: AsyncOpenAI,
        content: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        updated_content: list[dict[str, Any]] = []
        changed = False
        for part in content:
            new_part, part_changed = await self._normalize_input_part(client, part)
            updated_content.append(new_part)
            changed = changed or part_changed
        return updated_content, changed

    async def _normalize_output_item(
        self,
        client: AsyncOpenAI,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        output = item.get("output")
        if not isinstance(output, list):
            return item

        updated_output, changed = await self._normalize_content_parts(client, output)
        if not changed:
            return item

        normalized_item = dict(item)
        normalized_item["output"] = updated_output
        return normalized_item

    async def _normalize_message_item(
        self,
        client: AsyncOpenAI,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        content = item.get("content") or []
        updated_content, changed = await self._normalize_content_parts(client, content)
        if not changed:
            return item

        normalized_item = dict(item)
        normalized_item["content"] = updated_content
        return normalized_item

    async def _normalize_input_files(
        self, client: AsyncOpenAI, input_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in input_items:
            if item.get("type") in {"function_call_output", "custom_tool_call_output"}:
                normalized.append(await self._normalize_output_item(client, item))
                continue

            if item.get("type") != "message":
                normalized.append(item)
                continue
            normalized.append(await self._normalize_message_item(client, item))
        return normalized
