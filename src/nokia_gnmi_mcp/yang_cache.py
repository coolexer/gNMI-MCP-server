"""Local Nokia YANG path search from cached paths or checked-out YANG files."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger("nokia-gnmi-mcp.yang")


class YangSearch:
    """Search local Nokia YANG path caches, building them from submodules if present."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.yang_dir = self.project_root / "yang"
        self.cache_dir = self.yang_dir / "cache"
        self._cache: dict[str, list[str]] = {}

    def search(self, keyword: str, tree: str = "configure", max_results: int = 50) -> str:
        tree = tree.lower().strip()
        if tree not in ("configure", "state"):
            tree = "configure"
        max_results = max(1, min(max_results, 200))

        paths = self._load(tree)
        if not paths:
            return self._missing_cache_message(tree)

        keyword_lower = keyword.lower()
        matches = [path for path in paths if keyword_lower in path.lower()]
        if not matches:
            return f"No paths found containing '{keyword}' in /{tree} tree."

        total = len(matches)
        shown = matches[:max_results]
        result = f"Found {total} paths containing '{keyword}' in /{tree} tree"
        if total > max_results:
            result += f" (showing first {max_results})"
        return result + ":\n\n" + "\n".join(shown)

    def _missing_cache_message(self, tree: str) -> str:
        return (
            f"YANG cache not available for '{tree}'.\n\n"
            "For best results, use the separate nokia-yang-mcp server to find and "
            "validate paths before calling gNMI tools.\n\n"
            "To enable local fallback search, place Nokia YANG models under:\n"
            f"  {self.yang_dir}\n\n"
            "Expected structure:\n"
            "  yang/<release>/YANG/nokia-submodule/nokia-conf-*.yang\n"
            "  yang/<release>/YANG/nokia-submodule/nokia-state-*.yang\n\n"
            "Or pre-generate cache files:\n"
            "  yang/cache/configure-paths.txt\n"
            "  yang/cache/state-paths.txt"
        )

    def _load(self, tree: str) -> list[str]:
        if tree in self._cache:
            return self._cache[tree]

        cache_file = self.cache_dir / f"{tree}-paths.txt"
        if cache_file.exists():
            paths = [line for line in cache_file.read_text(encoding="utf-8").splitlines() if line]
            self._cache[tree] = paths
            return paths

        paths = self._build_from_yang(tree)
        if paths:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text("\n".join(paths), encoding="utf-8")
            self._cache[tree] = paths
        return paths

    def _find_submodule_dir(self, tree: str) -> Path | None:
        if not self.yang_dir.exists():
            return None
        pattern = "nokia-conf-*.yang" if tree == "configure" else "nokia-state-*.yang"
        for submod_dir in self.yang_dir.rglob("nokia-submodule"):
            if submod_dir.is_dir() and any(submod_dir.glob(pattern)):
                return submod_dir
        return None

    @staticmethod
    def _root_prefix(filename_stem: str, tree: str) -> str:
        base = "/configure" if tree == "configure" else "/state"
        file_prefix = "nokia-conf-" if tree == "configure" else "nokia-state-"
        rel = filename_stem.removeprefix(file_prefix)
        if rel == "router" or rel.startswith("router-"):
            return base + "/router[router-name]"
        if rel == "service" or rel.startswith("service-"):
            return base + "/service"
        if rel == "system" or rel.startswith("system-"):
            return base + "/system"
        return base

    def _build_from_yang(self, tree: str) -> list[str]:
        submod_dir = self._find_submodule_dir(tree)
        if not submod_dir:
            logger.warning("Nokia YANG submodule dir not found under %s", self.yang_dir)
            return []

        pattern = "nokia-conf-*.yang" if tree == "configure" else "nokia-state-*.yang"
        paths: set[str] = set()
        for yang_file in sorted(submod_dir.glob(pattern)):
            try:
                root = self._root_prefix(yang_file.stem, tree)
                self._extract_paths_from_file(yang_file, root, paths)
            except Exception as exc:
                logger.debug("Error parsing %s: %s", yang_file.name, exc)
        return sorted(paths)

    @staticmethod
    def _extract_paths_from_file(yang_file: Path, root_prefix: str, paths: set[str]) -> None:
        content = yang_file.read_text(encoding="utf-8", errors="ignore")
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        content = re.sub(r"//[^\n]*", "", content)
        stack: list[list[object]] = []

        for line in content.splitlines():
            container_match = re.match(r"^(\s*)container\s+([a-z][a-z0-9-]*)\s*\{", line)
            list_match = re.match(r"^(\s*)list\s+([a-z][a-z0-9-]*)\s*\{", line)
            key_match = re.match(r'^(\s*)key\s+"([^"]+)"', line)
            if not key_match:
                key_match = re.match(r"^(\s*)key\s+([a-z][a-z0-9-]*)\s*;", line)

            if container_match or list_match:
                is_list = bool(list_match)
                match = list_match or container_match
                assert match is not None
                indent = len(match.group(1))
                name = match.group(2)
                while stack and int(stack[-1][0]) >= indent:
                    stack.pop()
                stack.append([indent, name, is_list])
                path = root_prefix + "/" + "/".join(str(item[1]) for item in stack)
                if path != root_prefix:
                    paths.add(path)
            elif key_match and stack and bool(stack[-1][2]):
                key = key_match.group(2).split()[0]
                old = stack[-1]
                unkeyed = root_prefix + "/" + "/".join(str(item[1]) for item in stack)
                paths.discard(unkeyed)
                stack[-1] = [old[0], f"{old[1]}[{key}]", False]
                paths.add(root_prefix + "/" + "/".join(str(item[1]) for item in stack))
