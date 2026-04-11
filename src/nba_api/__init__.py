from importlib.metadata import PackageNotFoundError, version

name = "nba_api"
try:
    __version__ = version("nba_api")
except PackageNotFoundError:
    # Local dev: importing from source without `pip install -e .`
    __version__ = "0.0.0"
