#!/usr/bin/env python3
from __future__ import annotations

import base64
import http.client
import json
import os
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
USER_AGENT = "qdd-fuse-publisher"
MAX_GIT_BLOB_BYTES = 100 * 1024 * 1024


class GitHubError(RuntimeError):
    pass


def github_headers(token: str, content_type: str | None = "application/json") -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": API_VERSION,
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def request_json(
    token: str,
    method: str,
    url: str,
    payload: dict | None = None,
    expected: tuple[int, ...] = (200,),
    allow_404: bool = False,
) -> tuple[int, dict | list | None]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method, headers=github_headers(token))
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
            data = json.loads(raw.decode("utf-8")) if raw else None
            if response.status not in expected:
                raise GitHubError(f"{method} {url} returned {response.status}: {data}")
            return response.status, data
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        if allow_404 and exc.code == 404:
            return exc.code, None
        raise GitHubError(f"{method} {url} returned {exc.code}: {raw}") from exc


def ensure_repo(token: str, owner: str, name: str, private: bool) -> dict:
    status, repo = request_json(token, "GET", f"{API_ROOT}/repos/{owner}/{name}", allow_404=True)
    if status != 404:
        print(f"Using existing repository: {repo['html_url']}")
        return repo

    _, repo = request_json(
        token,
        "POST",
        f"{API_ROOT}/user/repos",
        {
            "name": name,
            "private": private,
            "auto_init": False,
            "description": "Reproducible QDD-Fuse package with checkpoint release assets.",
        },
        expected=(201,),
    )
    print(f"Created repository: {repo['html_url']}")
    return repo


def get_ref(token: str, owner: str, repo: str, branch: str) -> dict | None:
    try:
        status, ref = request_json(
            token,
            "GET",
            f"{API_ROOT}/repos/{owner}/{repo}/git/ref/heads/{branch}",
            allow_404=True,
        )
        return None if status == 404 else ref
    except GitHubError as exc:
        if "Git Repository is empty" in str(exc):
            return None
        raise


def initialize_empty_repo(token: str, owner: str, repo: str, branch: str) -> dict:
    content = base64.b64encode(b"Initializing QDD-Fuse repository.\n").decode("ascii")
    _, result = request_json(
        token,
        "PUT",
        f"{API_ROOT}/repos/{owner}/{repo}/contents/.init",
        {
            "message": "Initialize repository",
            "content": content,
            "branch": branch,
        },
        expected=(201,),
    )
    try:
        request_json(token, "PATCH", f"{API_ROOT}/repos/{owner}/{repo}", {"default_branch": branch})
    except GitHubError:
        print("Default-branch update was skipped; the token does not grant repository administration.")
    print(f"Initialized empty repository on {branch}.")
    return result


def source_files(package_dir: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    skipped: list[str] = []
    skip_exact = {
        "checkpoints/mosi/best.pt",
        "checkpoints/mosei/best.pt",
        "checkpoints/simsv2/best.pt",
    }
    skip_prefixes = ("data/", "outputs/", ".git/")

    for path in sorted(package_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(package_dir).as_posix()
        if rel in skip_exact or rel.startswith(skip_prefixes):
            skipped.append(rel)
            continue
        if path.stat().st_size >= MAX_GIT_BLOB_BYTES:
            skipped.append(rel)
            continue
        files.append(path)
    return files, skipped


def upload_source_tree(token: str, owner: str, repo: str, package_dir: Path, branch: str = "main") -> None:
    files, skipped = source_files(package_dir)
    ref = get_ref(token, owner, repo, branch)
    initialized_empty_repo = False
    if ref is None:
        initialize_empty_repo(token, owner, repo, branch)
        ref = get_ref(token, owner, repo, branch)
        initialized_empty_repo = True

    parent_sha = ref["object"]["sha"] if ref else None
    base_tree_sha = None
    preserve_existing = os.environ.get("PRESERVE_EXISTING", "false").lower() in {"1", "true", "yes"}
    if parent_sha and preserve_existing and not initialized_empty_repo:
        _, commit = request_json(token, "GET", f"{API_ROOT}/repos/{owner}/{repo}/git/commits/{parent_sha}")
        base_tree_sha = commit["tree"]["sha"]

    entries = []
    for index, path in enumerate(files, start=1):
        rel = path.relative_to(package_dir).as_posix()
        content = base64.b64encode(path.read_bytes()).decode("ascii")
        _, blob = request_json(
            token,
            "POST",
            f"{API_ROOT}/repos/{owner}/{repo}/git/blobs",
            {"content": content, "encoding": "base64"},
            expected=(201,),
        )
        mode = "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"
        entries.append({"path": rel, "mode": mode, "type": "blob", "sha": blob["sha"]})
        if index % 20 == 0 or index == len(files):
            print(f"Uploaded source blobs: {index}/{len(files)}")

    tree_payload = {"tree": entries}
    if base_tree_sha:
        tree_payload["base_tree"] = base_tree_sha
    _, tree = request_json(
        token,
        "POST",
        f"{API_ROOT}/repos/{owner}/{repo}/git/trees",
        tree_payload,
        expected=(201,),
    )
    commit_payload = {
        "message": "Add QDD-Fuse reproducibility package",
        "tree": tree["sha"],
        "parents": [parent_sha] if parent_sha else [],
    }
    _, commit = request_json(
        token,
        "POST",
        f"{API_ROOT}/repos/{owner}/{repo}/git/commits",
        commit_payload,
        expected=(201,),
    )
    if parent_sha:
        request_json(
            token,
            "PATCH",
            f"{API_ROOT}/repos/{owner}/{repo}/git/refs/heads/{branch}",
            {"sha": commit["sha"], "force": False},
        )
    else:
        request_json(
            token,
            "POST",
            f"{API_ROOT}/repos/{owner}/{repo}/git/refs",
            {"ref": f"refs/heads/{branch}", "sha": commit["sha"]},
            expected=(201,),
        )
        request_json(token, "PATCH", f"{API_ROOT}/repos/{owner}/{repo}", {"default_branch": branch})

    print(f"Committed {len(files)} source files to {branch}; skipped {len(skipped)} large/excluded files.")
    for rel in skipped:
        print(f"Skipped: {rel}")


def ensure_release(token: str, owner: str, repo: str, tag: str) -> dict:
    status, release = request_json(
        token,
        "GET",
        f"{API_ROOT}/repos/{owner}/{repo}/releases/tags/{tag}",
        allow_404=True,
    )
    if status != 404:
        print(f"Using existing release: {release['html_url']}")
        return release

    _, release = request_json(
        token,
        "POST",
        f"{API_ROOT}/repos/{owner}/{repo}/releases",
        {
            "tag_name": tag,
            "name": "QDD-Fuse reproducible package",
            "body": (
                "Full QDD-Fuse package split into tar parts. "
                "Reconstruct with: cat QDD-Fuse_repro_20260814.tar.part-* > QDD-Fuse.tar"
            ),
            "draft": False,
            "prerelease": False,
        },
        expected=(201,),
    )
    print(f"Created release: {release['html_url']}")
    return release


def delete_existing_asset(token: str, assets: list[dict], name: str) -> None:
    for asset in assets:
        if asset.get("name") == name:
            request_json(token, "DELETE", asset["url"], expected=(204,))
            print(f"Deleted existing release asset: {name}")
            return


def upload_asset(token: str, upload_url: str, path: Path) -> dict:
    base_url = upload_url.split("{", 1)[0]
    parsed = urllib.parse.urlparse(base_url)
    query = urllib.parse.urlencode({"name": path.name})
    target = f"{parsed.path}?{query}"
    size = path.stat().st_size

    conn = http.client.HTTPSConnection(parsed.netloc, timeout=1800)
    headers = github_headers(token, content_type="application/octet-stream")
    headers["Content-Length"] = str(size)
    conn.putrequest("POST", target)
    for key, value in headers.items():
        conn.putheader(key, value)
    conn.endheaders()

    sent = 0
    next_report = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            conn.send(chunk)
            sent += len(chunk)
            percent = int(sent * 100 / size) if size else 100
            if percent >= next_report:
                print(f"Uploading {path.name}: {percent}%")
                next_report += 10

    response = conn.getresponse()
    raw = response.read().decode("utf-8", errors="replace")
    conn.close()
    if response.status not in (200, 201):
        raise GitHubError(f"Asset upload failed for {path.name}: HTTP {response.status}: {raw}")
    return json.loads(raw)


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is required.", file=sys.stderr)
        return 2

    package_dir = Path(os.environ.get("PACKAGE_DIR", "/hy-tmp/QDD-Fuse")).resolve()
    asset_dir = Path(os.environ.get("ASSET_DIR", "/hy-tmp")).resolve()
    repo_name = os.environ.get("REPO_NAME", "QDD-Fuse")
    private = os.environ.get("PRIVATE", "true").lower() not in {"0", "false", "no"}
    release_tag = os.environ.get("RELEASE_TAG", "qdd-fuse-repro-20260814")

    if not package_dir.exists():
        raise SystemExit(f"Package directory does not exist: {package_dir}")

    _, user = request_json(token, "GET", f"{API_ROOT}/user")
    owner = os.environ.get("GITHUB_OWNER") or user["login"]
    print(f"Authenticated as: {owner}")

    repo = ensure_repo(token, owner, repo_name, private)
    upload_source_tree(token, owner, repo_name, package_dir)

    assets = sorted(asset_dir.glob("QDD-Fuse_repro_20260814.tar.part-*"))
    checksum = asset_dir / "QDD-Fuse_repro_20260814.tar.parts.sha256"
    if checksum.exists():
        assets.append(checksum)
    if not assets:
        raise SystemExit("No release assets found.")

    release = ensure_release(token, owner, repo_name, release_tag)
    _, release = request_json(token, "GET", f"{API_ROOT}/repos/{owner}/{repo_name}/releases/{release['id']}")
    for asset in assets:
        delete_existing_asset(token, release.get("assets", []), asset.name)
        print(f"Uploading release asset: {asset.name} ({asset.stat().st_size} bytes)")
        upload_asset(token, release["upload_url"], asset)
        time.sleep(1)
        _, release = request_json(token, "GET", f"{API_ROOT}/repos/{owner}/{repo_name}/releases/{release['id']}")

    print(f"Repository URL: {repo['html_url']}")
    print(f"Release URL: {release['html_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
