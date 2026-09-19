# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seeding HF_TOKEN from the notebook environment must replace a stale credential row.

A hosted runtime (Colab/Kaggle) starts with a fresh ephemeral ``UNSLOTH_STUDIO_HOME``, so the
credential encryption key in ``auth.db`` is new every session, while a persisted ``studio.db``
snapshot (``UNSLOTH_STUDIO_DB_BACKUP``) may be restored carrying an ``hf_token`` row encrypted
by a *previous* runtime's key. That row can never be decrypted again. ``colab.start()`` seeds
the notebook's ``HF_TOKEN`` secret, and the seed must replace such an unreadable row outright
(upsert) so the Studio UI sees a working token without a manual re-entry.
"""

import os
import sqlite3
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

_STALE_ROW = (
    "hf_token",
    "default",
    1,  # format_version matches, so only the undecryptable ciphertext marks it stale
)


@pytest.fixture()
def isolated_credential_db(monkeypatch, tmp_path):
    """A runtime-local studio.db path, no snapshot configured, no real auth.db touched."""
    from storage import credential_secrets

    live = tmp_path / "runtime" / "studio.db"
    monkeypatch.setattr(credential_secrets, "studio_db_path", lambda: live)
    # The key function is patched so the test never creates or reads a real auth.db: any fixed
    # 32-byte key behaves like "the key this runtime generated", keeping the test hermetic.
    monkeypatch.setattr(
        credential_secrets, "get_or_create_credential_encryption_key", lambda: bytes(range(32))
    )
    monkeypatch.setenv("UNSLOTH_STUDIO_DB_BACKUP", str(tmp_path / "unused-snapshot.db"))
    credential_secrets.reset_schema_state_for_tests()
    yield live
    credential_secrets.reset_schema_state_for_tests()


def _insert_row_encrypted_by_a_previous_key(credential_secrets, live):
    """Simulate a Drive-restored row: valid shape, but undecryptable with any current key."""
    conn = credential_secrets.get_connection()  # creates the schema, as the first caller would
    conn.close()
    with sqlite3.connect(live) as conn:
        conn.execute(
            """
            INSERT INTO credential_secrets (
                credential_kind, scope_id, format_version,
                nonce, ciphertext, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 't0', 't0')
            """,
            (_STALE_ROW[0], _STALE_ROW[1], _STALE_ROW[2], os.urandom(12), os.urandom(48)),
        )


def test_seed_replaces_stale_restored_token(monkeypatch, isolated_credential_db):
    from colab import _seed_hf_token_from_environment
    from storage import credential_secrets

    _insert_row_encrypted_by_a_previous_key(credential_secrets, isolated_credential_db)
    monkeypatch.setenv("HF_TOKEN", "notebook-secret-token")

    _seed_hf_token_from_environment()

    assert credential_secrets.get_secret("hf_token", "default") == "notebook-secret-token"


def test_seed_without_environment_leaves_database_untouched(monkeypatch, isolated_credential_db):
    from colab import _seed_hf_token_from_environment
    from storage import credential_secrets

    _insert_row_encrypted_by_a_previous_key(credential_secrets, isolated_credential_db)
    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(env_var, raising = False)
    before = Path(isolated_credential_db).read_bytes()

    _seed_hf_token_from_environment()

    # No environment token: the seeding is a no-op, so the stale row stays exactly as restored.
    assert Path(isolated_credential_db).read_bytes() == before


def test_seed_replaces_even_a_readable_token(monkeypatch, isolated_credential_db):
    """At session start the notebook secret is the source of truth: a readable row that came
    from a restored snapshot is still replaced by the environment token."""
    from colab import _seed_hf_token_from_environment
    from storage import credential_secrets

    credential_secrets.save_hf_token("token-entered-in-the-ui")
    monkeypatch.setenv("HF_TOKEN", "notebook-secret-token")

    _seed_hf_token_from_environment()

    assert credential_secrets.get_secret("hf_token", "default") == "notebook-secret-token"
