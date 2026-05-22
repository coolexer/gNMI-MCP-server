from __future__ import annotations

from nokia_gnmi_mcp.yang_cache import YangSearch


def test_yang_search_missing_tree_returns_helpful_message(tmp_path):
    search = YangSearch(project_root=tmp_path)

    result = search.search("bgp", tree="configure", max_results=3)

    assert "YANG cache not available" in result
    assert str(tmp_path / "yang") in result
    assert "_YANG_DIR" not in result


def test_yang_search_reads_existing_cache(tmp_path):
    cache_dir = tmp_path / "yang" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "configure-paths.txt").write_text(
        "\n".join(
            [
                "/configure/router[router-name]/bgp",
                "/configure/router[router-name]/interface",
                "/configure/service/vprn[service-name]",
            ]
        ),
        encoding="utf-8",
    )
    search = YangSearch(project_root=tmp_path)

    result = search.search("router", tree="configure", max_results=1)

    assert "Found 2 paths" in result
    assert "(showing first 1)" in result
    assert "/configure/router[router-name]/bgp" in result
