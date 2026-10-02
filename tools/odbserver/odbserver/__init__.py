import sys

try:
    if sys.version_info[:2] >= (3, 8):
        from importlib.metadata import PackageNotFoundError, version
        __version__ = version("pylife-odbserver")
    else:
        import pkg_resources
        __version__ = pkg_resources.get_distribution("pylife-odbserver").version
except Exception:
    __version__ = "unknown"
