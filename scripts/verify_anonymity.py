#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from PIL import Image
except Exception:
    Image = None

TEXT_SUFFIXES = {
    '.py', '.sh', '.md', '.txt', '.json', '.jsonl', '.yaml', '.yml', '.csv',
    '.tex', '.bib', '.toml', '.ini', '.cfg', '.xml', '.rst'
}
CHUNK_SIZE = 1024 * 1024
OVERLAP = 512

EMAIL_RE = re.compile(
    rb'\b[A-Z0-9._%+-]+@(?!example\.(?:com|org|net)\b|localhost\b)'
    rb'[A-Z0-9.-]+\.[A-Z]{2,}\b',
    re.IGNORECASE,
)
IDENTITY_DECL_RE = re.compile(
    rb'\b(?:author|creator|maintainer|affiliation|orcid|corresponding[ _-]?author|'
    rb'created[ _-]?by|written[ _-]?by)\b\s*[:=]',
    re.IGNORECASE,
)
RISKY_FILENAMES = re.compile(
    r'(?i)(id_rsa|id_ed25519|credentials?\.json|service[_-]?account|\.env(?:\.|$)|'
    r'secrets?\.|authorized_keys|known_hosts|\.ds_store$|thumbs\.db$)'
)
RISKY_PATH_PARTS = {
    '.git', '.svn', '.hg', '.idea', '.vscode', '.ipynb_checkpoints', '__MACOSX'
}
ALLOWED_HIDDEN = {'.gitignore'}

REPO_MARKERS = (b'github.com/', b'gitlab.com/', b'bitbucket.com/', b'bitbucket.org/')
PRIVATE_PATH_MARKERS = (
    b'/root/', b'/home/', b'/users/', b'/mnt/data/', b'/workspace/',
    b'/workspaces/', b'/content/drive/', b'/autodl-tmp/',
    b'c:\\users\\', b'c:\\documents and settings\\', b'c:\\desktop\\',
    b'c:\\downloads\\', b'c:\\onedrive\\', b'f:\\drug\\',
)
SECRET_MARKERS = (
    b'-----begin rsa private key-----',
    b'-----begin ec private key-----',
    b'-----begin openssh private key-----',
)
API_KEY_LABELS = (
    b'openai_api_key', b'anthropic_api_key', b'google_api_key',
    b'gemini_api_key', b'hf_api_key', b'huggingface_api_key',
)
LEGACY_ALIASES = (b'medguard-geriatrics', b'shelby')
IDENTITY_JSON_KEYS = (
    b'"author_name"', b"'author_name'", b'"author_email"', b"'author_email'",
    b'"affiliation"', b"'affiliation'", b'"patient_name"', b"'patient_name'",
    b'"full_name"', b"'full_name'", b'"medical_record_number"',
    b"'medical_record_number'", b'"mrn"', b"'mrn'", b'"street_address"',
    b"'street_address'", b'"phone_number"', b"'phone_number'",
)


def excerpt(data: bytes, index: int, length: int = 120) -> str:
    return data[index:index + length].decode('utf-8', errors='replace')


def scan_bytes(path: Path, is_text: bool) -> tuple[str, str] | None:
    tail = b''
    try:
        with path.open('rb') as stream:
            while True:
                block = stream.read(CHUNK_SIZE)
                if not block:
                    return None
                data = tail + block
                lower = data.lower()

                if b'@' in data:
                    match = EMAIL_RE.search(data)
                    if match:
                        return 'personal_email', match.group(0)[:120].decode(
                            'utf-8', errors='replace'
                        )

                for marker in REPO_MARKERS:
                    index = lower.find(marker)
                    if index >= 0:
                        after = lower[index + len(marker):index + len(marker) + 32]
                        if not (
                            after.startswith(b'example')
                            or after.startswith(b'anonymous')
                            or after.startswith(b'anon-')
                            or after.startswith(b'anon/')
                        ):
                            return 'personal_repo', excerpt(data, index)

                for marker in PRIVATE_PATH_MARKERS:
                    index = lower.find(marker)
                    if index >= 0:
                        return 'private_path', excerpt(data, index)

                index = lower.find(b'file:///')
                if index >= 0:
                    return 'local_build_uri', excerpt(data, index)

                for marker in SECRET_MARKERS:
                    index = lower.find(marker)
                    if index >= 0:
                        return 'private_key', excerpt(data, index)

                index = lower.find(b'sk-')
                if index >= 0:
                    token = lower[index:index + 96]
                    if re.match(rb'sk-[a-z0-9_-]{16,}', token):
                        return 'api_secret', excerpt(data, index)

                for label in API_KEY_LABELS:
                    index = lower.find(label)
                    if index >= 0:
                        window = lower[index:index + 240]
                        if b'=' in window or b':' in window:
                            return 'api_secret', excerpt(data, index)

                for alias in LEGACY_ALIASES:
                    index = lower.find(alias)
                    if index >= 0:
                        return 'legacy_project_alias', excerpt(data, index)

                if is_text:
                    for key in IDENTITY_JSON_KEYS:
                        index = lower.find(key)
                        if index >= 0:
                            window = lower[index:index + len(key) + 16]
                            if b':' in window:
                                return 'identity_json_key', excerpt(data, index)
                    if any(
                        marker in lower
                        for marker in (
                            b'author', b'creator', b'maintainer', b'affiliation',
                            b'orcid', b'created_by', b'written_by'
                        )
                    ):
                        match = IDENTITY_DECL_RE.search(data)
                        if match:
                            return 'identity_declaration', match.group(0)[:120].decode(
                                'utf-8', errors='replace'
                            )

                tail = data[-OVERLAP:]
    except OSError:
        return None


def scan_docx(path: Path, root: Path, hits: list[tuple[str, str, str]]) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if any(name.startswith('customXml/') for name in names):
                hits.append(('docx_custom_xml', str(path.relative_to(root)), 'customXml/'))
            for name in ('docProps/core.xml', 'docProps/app.xml', 'docProps/custom.xml'):
                if name not in names:
                    continue
                tree = ET.fromstring(archive.read(name))
                sensitive_fields = {
                    'creator', 'lastModifiedBy', 'Company', 'Manager', 'Author',
                    'LastSavedBy', 'HyperlinkBase'
                }
                for element in tree.iter():
                    local = element.tag.rsplit('}', 1)[-1]
                    value = (element.text or '').strip()
                    if local in sensitive_fields and value:
                        hits.append((
                            'docx_identity_metadata', str(path.relative_to(root)),
                            f'{local}={value[:100]}'
                        ))
    except Exception as exc:
        hits.append(('invalid_docx', str(path.relative_to(root)), str(exc)[:100]))


def scan_png(path: Path, root: Path, hits: list[tuple[str, str, str]]) -> None:
    if Image is None:
        return
    try:
        with Image.open(path) as image:
            identity_keys = {
                'Author', 'author', 'Creator', 'creator', 'Artist', 'artist',
                'Copyright', 'copyright', 'Comment', 'comment', 'Description',
                'description', 'Software', 'software'
            }
            for key, value in image.info.items():
                if key in identity_keys and str(value).strip():
                    hits.append((
                        'png_identity_metadata', str(path.relative_to(root)),
                        f'{key}={str(value)[:100]}'
                    ))
    except Exception as exc:
        hits.append(('invalid_png', str(path.relative_to(root)), str(exc)[:100]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    hits: list[tuple[str, str, str]] = []

    for path in root.rglob('*'):
        rel = path.relative_to(root)
        if path.is_symlink():
            hits.append(('symlink', str(rel), str(path.readlink())[:100]))
            continue
        if any(part in RISKY_PATH_PARTS for part in rel.parts):
            hits.append(('vcs_or_editor_path', str(rel), 'risky path component'))
        if any(part.startswith('.') and part not in ALLOWED_HIDDEN for part in rel.parts):
            hits.append(('hidden_path', str(rel), 'hidden path component'))
        if path.is_file() and RISKY_FILENAMES.search(path.name):
            hits.append(('risky_filename', str(rel), path.name))
        if not path.is_file():
            continue
        if rel.as_posix() == 'scripts/verify_anonymity.py':
            continue

        finding = scan_bytes(path, path.suffix.lower() in TEXT_SUFFIXES)
        if finding:
            hits.append((finding[0], str(rel), finding[1]))

        if path.suffix.lower() == '.docx':
            scan_docx(path, root, hits)
        elif path.suffix.lower() == '.png':
            scan_png(path, root, hits)

    if hits:
        print('[WARNING] Potential anonymity findings:')
        for item in hits:
            print(' -', item)
        if args.strict:
            raise SystemExit(1)
    else:
        print('[OK] Strict release scan found no direct identity, identifying repository link, private path, secret, legacy project alias, risky metadata, hidden/VCS file, or symlink.')


if __name__ == '__main__':
    main()
