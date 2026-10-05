"""Tests for the deferred image decoding a VLM dataset can carry.

The module is loaded straight from its file so the walk can be tested without
importing `unsloth` (which pulls torch and the GPU init) - it depends on nothing
but the standard library, and that is worth asserting rather than assuming.
"""

import importlib.util
import io
import pathlib

import pytest

MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "unsloth" / "utils" / "lazy_image.py"


@pytest.fixture(scope = "module")
def lazy():
    spec = importlib.util.spec_from_file_location("_unsloth_lazy_image_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(image):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image", "image": image},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "there"}]},
        ]
    }


def test_the_module_needs_nothing_but_the_standard_library(lazy):
    assert lazy.__file__ == str(MODULE_PATH)
    assert lazy.LazyImage.__module__ == lazy.__name__


def test_a_bytes_cell_is_deferred(lazy):
    wrapped = lazy.LazyImage.from_value({"bytes": b"jpeg-bytes", "path": None})
    assert lazy.is_lazy_image(wrapped)
    assert wrapped.bytes == b"jpeg-bytes"
    assert wrapped.path is None


def test_a_path_only_cell_keeps_the_path(lazy):
    wrapped = lazy.LazyImage.from_value({"bytes": None, "path": "/some/where.jpg"})
    assert wrapped.path == "/some/where.jpg"
    assert wrapped.bytes is None


def test_a_string_path_is_deferred_and_asks_for_rgb(lazy, tmp_path):
    # The eager path received this cell as a string and converted it to RGB, so
    # the deferred decode has to as well or the pixels would differ.
    path = tmp_path / "shot.jpg"
    path.write_bytes(b"not really a jpeg")
    wrapped = lazy.LazyImage.from_value(str(path))
    assert wrapped.path == str(path)
    assert wrapped.to_rgb is True


def test_a_cell_datasets_already_decoded_keeps_its_mode(lazy):
    # datasets.Image.decode_example never converts colour, so an eager run trained
    # on those pixels as-is; forcing RGB here would change what the model sees.
    assert lazy.LazyImage.from_value({"bytes": b"x", "path": None}).to_rgb is False


def test_a_missing_path_is_not_deferred(lazy, tmp_path):
    # Left to the caller's eager branch, which reports it exactly as before.
    assert lazy.LazyImage.from_value(str(tmp_path / "gone.jpg")) is None


@pytest.mark.parametrize(
    "remote",
    ["https://host/a.jpg", "http://host/a.jpg", "data:image/png;base64,AA"],
)
def test_a_remote_string_is_not_deferred(lazy, remote):
    assert lazy.LazyImage.from_value(remote) is None


def test_already_deferred_values_pass_through(lazy):
    once = lazy.LazyImage.from_value({"bytes": b"x", "path": None})
    assert lazy.LazyImage.from_value(once) is once


def test_resolving_decodes_only_the_image_parts(lazy, monkeypatch):
    monkeypatch.setattr(lazy.LazyImage, "to_pil", lambda self: f"decoded:{self.bytes!r}")
    deferred = lazy.LazyImage(bytes = b"jpeg", to_rgb = False)

    batch = [_row(deferred)]

    resolved = lazy.resolve_lazy_images(batch)
    content = resolved[0]["messages"][0]["content"]

    assert content[0] == {"type": "text", "text": "hi"}
    assert content[1]["image"] == "decoded:b'jpeg'"
    # The rest of the turn is shared, not copied.
    assert resolved[0]["messages"][1] is batch[0]["messages"][1]


def test_resolving_leaves_the_dataset_row_alone(lazy, monkeypatch):
    # Rows are fetched again every epoch: mutating this one would turn the first
    # batch's decode into the object every later batch reuses.
    monkeypatch.setattr(lazy.LazyImage, "to_pil", lambda self: "decoded")
    deferred = lazy.LazyImage(bytes = b"jpeg")

    row = _row(deferred)
    lazy.resolve_lazy_images([row])

    assert row["messages"][0]["content"][1]["image"] is deferred


def test_a_batch_without_deferred_images_is_returned_untouched(lazy):
    batch = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}]

    resolved = lazy.resolve_lazy_images(batch)

    assert resolved[0] is batch[0]
    assert resolved[0]["messages"] is batch[0]["messages"]


def test_conversations_are_resolved_too(lazy, monkeypatch):
    monkeypatch.setattr(lazy.LazyImage, "to_pil", lambda self: "decoded")
    row = {
        "conversations": [
            {"role": "user", "content": [{"type": "image", "image": lazy.LazyImage(bytes = b"x")}]}
        ]
    }

    resolved = lazy.resolve_lazy_images([row])

    assert resolved[0]["conversations"][0]["content"][0]["image"] == "decoded"


def test_a_video_part_is_never_touched(lazy, monkeypatch):
    monkeypatch.setattr(lazy.LazyImage, "to_pil", lambda self: "decoded")
    row = {
        "messages": [
            {"role": "user", "content": [{"type": "video", "video": lazy.LazyImage(bytes = b"x")}]}
        ]
    }

    resolved = lazy.resolve_lazy_images([row])

    assert resolved[0] is row


def test_the_decode_reproduces_the_eager_one(lazy):
    Image = pytest.importorskip("PIL.Image")

    source = Image.new("RGBA", (4, 3), (10, 20, 30, 128))
    buffer = io.BytesIO()
    source.save(buffer, format = "PNG")
    raw = buffer.getvalue()

    as_is = lazy.LazyImage(bytes = raw).to_pil()
    assert as_is.mode == Image.open(io.BytesIO(raw)).mode
    assert as_is.tobytes() == Image.open(io.BytesIO(raw)).tobytes()

    converted = lazy.LazyImage(bytes = raw, to_rgb = True).to_pil()
    assert converted.mode == "RGB"
    assert converted.tobytes() == Image.open(io.BytesIO(raw)).convert("RGB").tobytes()


def test_the_decode_of_a_path_reproduces_the_eager_one(lazy, tmp_path):
    Image = pytest.importorskip("PIL.Image")

    path = tmp_path / "shot.png"
    Image.new("L", (5, 2), 7).save(path, format = "PNG")

    decoded = lazy.LazyImage(path = str(path), to_rgb = True).to_pil()

    assert decoded.mode == "RGB"
    assert decoded.tobytes() == Image.open(path).convert("RGB").tobytes()