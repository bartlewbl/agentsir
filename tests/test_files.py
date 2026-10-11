"""File tools stay inside the workspace and can page through an upload."""

import pytest

import tools.files as files
from tools.files import (
    describe_upload,
    forget_uploads,
    list_files,
    read_file,
    remember_upload,
    search_files,
)

PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 68>>stream
BT /F1 24 Tf 20 100 Td (Hello from a file) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000052 00000 n 
0000000101 00000 n 
0000000229 00000 n 
0000000346 00000 n 
trailer<</Size 6/Root 1 0 R>>
startxref
419
%%EOF
"""


@pytest.fixture(autouse=True)
def workspace(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("AGENT_FILES_ROOT", str(root))
    files._uploads.clear()
    files._by_user.clear()
    files._cached_text.cache_clear()
    yield root
    files._uploads.clear()
    files._by_user.clear()
    files._cached_text.cache_clear()


def test_list_search_and_read_a_file(workspace):
    (workspace / "notes.txt").write_text("the budget is late\n")
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("print('budget')\n")

    listing = list_files(".")
    assert "notes.txt" in listing
    assert "src/" in listing

    found = search_files("budget", "src")
    assert "src/app.py:1:" in found
    assert "print('budget')" in found

    assert "the budget is late" in read_file("notes.txt")
    assert read_file("notes.txt").startswith("File: notes.txt\nPart 1 of 1")


def test_long_files_come_in_parts():
    file_id = remember_upload(7, "big.txt", ("row\n" * 4000).encode())
    note = describe_upload(file_id, "What repeats?")

    assert file_id in note
    assert "What repeats?" in note
    first = read_file(file_id, part=1)
    assert "Part 1 of 1" not in first
    assert "row" in first
    assert "row" in read_file(file_id, part=2)

    forget_uploads(7)
    assert "send it again" in read_file(file_id)


def test_old_uploads_are_dropped():
    ids = [remember_upload(3, f"{i}.txt", f"number {i}\n".encode()) for i in range(9)]

    assert "send it again" in read_file(ids[0])
    assert "number 8" in read_file(ids[-1])


def test_pdfs_are_read(workspace):
    (workspace / "hello.pdf").write_bytes(PDF)

    assert "Hello from a file" in read_file("hello.pdf")
    file_id = remember_upload(1, "hello.pdf", PDF)
    assert "Hello from a file" in read_file(file_id)


def test_secrets_and_paths_outside_the_workspace_are_refused(workspace, tmp_path):
    (workspace / ".env").write_text("OPENAI_API_KEY=sk-secret\n")
    outside = tmp_path / "secret.txt"
    outside.write_text("hidden")
    link = workspace / "link.txt"
    link.symlink_to(outside)

    (workspace / ".env.example").write_text("OPENAI_API_KEY=\n")
    assert "OPENAI_API_KEY=" in read_file(".env.example")
    assert "sk-secret" not in read_file(".env")
    assert "secret" in read_file(".env")
    assert search_files("sk-secret").startswith("No matches")
    found = search_files("OPENAI_API_KEY")
    assert ".env.example" in found
    assert "sk-secret" not in found
    assert ".env" not in list_files(".").splitlines()
    assert "hidden" not in read_file(str(outside))
    assert "outside" in read_file(str(outside))
    assert "hidden" not in read_file("link.txt")
    assert "outside" in read_file("link.txt")


def test_unreadable_uploads_are_refused():
    with pytest.raises(ValueError, match="empty"):
        remember_upload(1, "empty.txt", b"")
    with pytest.raises(ValueError, match="isn't one of those"):
        remember_upload(1, "pic.bin", b"\x00\x01\x02\xff" * 20)
