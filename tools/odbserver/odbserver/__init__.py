import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("pylife-odbserver").version
except:
    __version__ = 'unknown'
