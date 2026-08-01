"""Task implementations and Gym environment registrations for frog_lab."""

from isaaclab_tasks.utils import import_packages

# Import task configuration packages so their Gym registrations are available
# when the extension package is imported.
_BLACKLIST_PKGS = ["utils", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
