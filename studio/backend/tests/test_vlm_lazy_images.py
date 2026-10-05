"""Vision dataset conversion must be able to defer image decoding.

Decoding the whole dataset at load time is what turns a 1.7GB dataset into tens of
gigabytes of RGB buffers and kills the worker with a RAM OOM before the first step.
These tests pin both halves of the fix: the conversion keeps images compressed, and
the collator's deferred decode returns the exact pixels the eager path produced.
"""

import io
import pathlib

import pytest

datasets = pytest.importorskip("datasets")
Image = pytest.importorskip("PIL.Image")

from utils.datasets.format_conversion import (  # noqa: E402
    convert_llava_to_vlm_format,
    convert_sharegpt_with_images_to_vlm_format,
    convert_to_vlm_format,
)
from utils.datasets.vlm_lazy import (  # noqa: E402
    undecoded_image_columns,
    wrap_lazy_images_in_messages,
)

TRAINER_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1] / "core" / "training" / "trainer.py"
)


@pytest.fixture(scope = "module")
def LazyImage():
    return pytest.importorskip("unsloth.utils.lazy_image").LazyImage


def _png_bytes(size = (6, 4), colour = (10, 20, 30, 255)):
    buffer = io.BytesIO()
    Image.new("RGBA", size, colour).save(buffer, format = "PNG")
    return buffer.getvalue()


def _image_text_dataset(rows):
    features = datasets.Features(
        {"image": datasets.Image(), "text": datasets.Value("string")}
    )
    return datasets.Dataset.from_list(rows, features = features)


def _first_image(row):
    return row["messages"][0]["content"][1]["image"]


def test_the_default_conversion_still_decodes_up_front(LazyImage):
    dataset = _image_text_dataset([{"image": {"bytes": _png_bytes(), "path": None}, "text": "a"}])

    converted = convert_to_vlm_format(dataset, instruction = "Describe")

    image = _first_image(converted[0])
    assert not isinstance(image, LazyImage)
    assert isinstance(image, Image.Image)


def test_the_lazy_conversion_keeps_the_image_compressed(LazyImage):
    raw = _png_bytes()
    dataset = _image_text_dataset([{"image": {"bytes": raw, "path": None}, "text": "a"}])

    converted = convert_to_vlm_format(
        dataset, instruction = "Describe", lazy_images = True
    )

    image = _first_image(converted[0])
    assert isinstance(image, LazyImage)
    assert image.bytes == raw
    # No colour conversion: datasets decodes those cells without one, so the
    # pixels an eager run trained on have to be reproduced as they were.
    assert image.to_rgb is False


def test_the_deferred_decode_is_pixel_identical_to_the_eager_one(LazyImage):
    raw = _png_bytes(size = (8, 5))
    dataset = _image_text_dataset([{"image": {"bytes": raw, "path": None}, "text": "a"}])

    eager = convert_to_vlm_format(dataset, instruction = "Describe")
    lazy = convert_to_vlm_format(dataset, instruction = "Describe", lazy_images = True)

    eager_image = _first_image(eager[0])
    decoded = _first_image(lazy[0]).to_pil()

    assert decoded.mode == eager_image.mode
    assert decoded.tobytes() == eager_image.tobytes()


def test_an_image_folder_cell_is_deferred_as_a_path(LazyImage, tmp_path):
    path = tmp_path / "shot.png"
    path.write_bytes(_png_bytes())
    dataset = _image_text_dataset([{"image": {"bytes": None, "path": str(path)}, "text": "a"}])

    converted = convert_to_vlm_format(
        dataset, instruction = "Describe", lazy_images = True
    )

    image = _first_image(converted[0])
    assert isinstance(image, LazyImage)
    assert image.path == str(path)
    # A path cell reached the eager path as a string, which it converted to RGB.
    assert image.to_rgb is True
    assert image.to_pil().tobytes() == Image.open(path).convert("RGB").tobytes()


def test_a_string_column_is_left_alone(tmp_path):
    # Casting a Value("string") column to Image would rewrite each cell as
    # {"path": ..., "bytes": None} and hide the URL-versus-local-path branching.
    path = tmp_path / "shot.png"
    path.write_bytes(_png_bytes())
    dataset = datasets.Dataset.from_dict({"image": [str(path)], "text": ["a"]})

    unchanged = undecoded_image_columns(dataset, ["image"])

    assert unchanged[0]["image"] == str(path)


def test_sharegpt_conversion_defers_and_still_matches(LazyImage):
    raw = _png_bytes(size = (7, 3))
    features = datasets.Features(
        {"image": datasets.Image(), "conversations": datasets.Value("string")}
    )
    dataset = datasets.Dataset.from_list(
        [{"image": {"bytes": raw, "path": None}, "conversations": "<image> What is this?"}],
        features = features,
    )

    eager = convert_sharegpt_with_images_to_vlm_format(dataset)
    lazy = convert_sharegpt_with_images_to_vlm_format(dataset, lazy_images = True)

    eager_image = eager[0]["messages"][0]["content"][1]["image"]
    lazy_image = lazy[0]["messages"][0]["content"][1]["image"]
    assert isinstance(lazy_image, LazyImage)
    assert lazy_image.to_pil().mode == eager_image.mode
    assert lazy_image.to_pil().tobytes() == eager_image.tobytes()


def _llava_row(image):
    return {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ],
        "images": [image],
    }


def test_llava_conversion_defers_and_still_matches(LazyImage):
    raw = _png_bytes(size = (4, 4))

    eager = convert_llava_to_vlm_format([_llava_row(Image.open(io.BytesIO(raw)))])
    lazy = convert_llava_to_vlm_format(
        [_llava_row({"bytes": raw, "path": None})], lazy_images = True
    )

    eager_image = eager[0]["messages"][0]["content"][0]["image"]
    lazy_image = lazy[0]["messages"][0]["content"][0]["image"]

    assert isinstance(lazy_image, LazyImage)
    assert lazy_image.to_pil().mode == eager_image.mode
    assert lazy_image.to_pil().tobytes() == eager_image.tobytes()


def test_the_messages_branch_defers_nested_images(LazyImage):
    raw = _png_bytes(size = (5, 5))
    features = datasets.Features(
        {
            "messages": [
                {
                    "role": datasets.Value("string"),
                    "content": [
                        {
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "image": datasets.Image(),
                        }
                    ],
                }
            ]
        }
    )
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image", "image": {"bytes": raw, "path": None}},
                ],
            }
        ]
    }
    try:
        dataset = datasets.Dataset.from_list([row], features = features)
    except Exception as error:  # pragma: no cover - depends on the datasets release
        pytest.skip(f"This datasets release cannot nest an Image in a struct: {error}")

    decoded_eager = dataset[0]["messages"][0]["content"][1]["image"]
    wrapped = wrap_lazy_images_in_messages(dataset)

    image = wrapped[0]["messages"][0]["content"][1]["image"]
    assert isinstance(image, LazyImage)
    assert image.to_pil().mode == decoded_eager.mode
    assert image.to_pil().tobytes() == decoded_eager.tobytes()
    # The text part is shared, not rebuilt.
    assert wrapped[0]["messages"][0]["content"][0]["text"] == "hi"


def test_a_row_of_decoded_images_is_left_untouched(LazyImage):
    row = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "image": Image.new("RGB", (2, 2))}],
            }
        ]
    }

    assert wrap_lazy_images_in_messages([row])[0] is row


def test_the_training_path_defers_images():
    # The GPU training worker is what OOMed; the MLX worker and the dataset
    # preview keep the eager path, so this assertion is on the call site.
    source = TRAINER_SOURCE.read_text(encoding = "utf-8")
    train_call = source[source.index("dataset_info = format_and_template_dataset(") :][:800]
    eval_call = source[source.index("eval_info = format_and_template_dataset(") :][:600]

    assert "lazy_images = True" in train_call
    assert "lazy_images = True" in eval_call