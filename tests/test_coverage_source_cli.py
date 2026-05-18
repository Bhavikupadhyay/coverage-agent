from coverage_agent.sandbox.e2b_runner import coverage_source_cli_flag


def test_slugify_package():
    assert "slugify" in coverage_source_cli_flag("slugify/__main__.py")


def test_skips_src_layout_prefix():
    assert "requests" in coverage_source_cli_flag("src/requests/auth.py")


def test_empty_path():
    assert coverage_source_cli_flag("") == ""
