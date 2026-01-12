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

    async def _normalize_input_files(
        self, client: AsyncOpenAI, input_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in input_items:
            if item.get("type") != "message":
                normalized.append(item)
                continue
            content = item.get("content") or []
            updated_content: list[dict[str, Any]] = []
            changed = False
            for part in content:
                if part.get("type") != "input_file":
                    if part.get("type") == "input_image":
                        detail = part.get("detail")
                        image_url = part.get("image_url")
                        file_id = part.get("file_id")
                        if file_id:
                            new_part = {"type": "input_image", "file_id": file_id}
                            if detail:
                                new_part["detail"] = detail
                            updated_content.append(new_part)
                            if image_url:
                                changed = True
                            continue
                        if image_url and image_url.startswith("file://"):
                            local_path = Path(image_url[len("file://") :])
                            if local_path.exists():
                                data_bytes = local_path.read_bytes()
                                mime_type = guess_mime_type(local_path.name)
                                file_id = await self._upload_file_bytes(
                                    client, data_bytes, local_path.name, mime_type
                                )
                                new_part = {"type": "input_image", "file_id": file_id}
                                if detail:
                                    new_part["detail"] = detail
                                updated_content.append(new_part)
                                changed = True
                                continue
                    updated_content.append(part)
                    continue
                if part.get("file_id"):
                    updated_content.append(
                        {"type": "input_file", "file_id": part.get("file_id")}
                    )
                    if part.get("filename") or part.get("file_url") or part.get("file_data"):
                        changed = True
                    continue

                filename = part.get("filename")
                file_data = part.get("file_data")
                file_url = part.get("file_url")

                if file_data:
                    data_bytes, detected_mime = self._decode_file_data(file_data)
                    if data_bytes is None:
                        updated_content.append(part)
                        continue
                    mime_type = detected_mime or (guess_mime_type(filename) if filename else None)
                    file_id = await self._upload_file_bytes(
                        client, data_bytes, filename, mime_type
                    )
                    updated_content.append({"type": "input_file", "file_id": file_id})
                    changed = True
                    continue

                if file_url:
                    mime_type = None
                    data_bytes = None
                    if file_url.startswith("data:"):
                        data_bytes, mime_type = self._decode_file_data(file_url)
                    elif file_url.startswith("file://"):
                        local_path = Path(file_url[len("file://") :])
                        if local_path.exists():
                            data_bytes = local_path.read_bytes()
                            if not filename:
                                filename = local_path.name
                            mime_type = guess_mime_type(local_path.name)
                    if data_bytes is not None:
                        file_id = await self._upload_file_bytes(
                            client, data_bytes, filename, mime_type
                        )
                        updated_content.append({"type": "input_file", "file_id": file_id})
                        changed = True
                        continue

                updated_content.append(part)

            if changed:
                item = dict(item)
                item["content"] = updated_content
            normalized.append(item)
        return normalized
