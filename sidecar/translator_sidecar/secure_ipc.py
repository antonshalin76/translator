"""Open the private IPC directory without traversing symlinks."""

import os
import stat
from pathlib import Path


def open_private_directory(path: Path) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise PermissionError("socket parent must be an absolute directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(path.anchor, flags)
    try:
        for name in path.parts[1:]:
            try:
                child = os.open(name, flags, dir_fd=descriptor)
            except OSError as error:
                entry = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if stat.S_ISLNK(entry.st_mode):
                    raise PermissionError(
                        "socket parent must be a real directory"
                    ) from error
                raise
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.getuid():
            raise PermissionError("socket parent owner mismatch")
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise PermissionError("socket parent mode must be 0700")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise
