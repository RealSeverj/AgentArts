"""Publish an immutable best-checkpoint snapshot without serializing it twice."""
from __future__ import annotations

import errno
import os
from pathlib import Path
import shutil
import uuid


_LINK_FALLBACK_ERRNOS = {errno.EXDEV, errno.ENOSYS, errno.EPERM,
                         getattr(errno, 'ENOTSUP', -1), getattr(errno, 'EOPNOTSUPP', -1)}
_LINK_FALLBACK_WINERRORS = {1, 17, 50, 1314}


def publish_checkpoint_snapshot(source, destination):
    """Atomically publish source as destination; return ``hardlink`` or ``copy``.

    ``source`` must already have been published by atomic save-and-replace.
    Its contents must never subsequently be edited in place: a later atomic
    replacement of last.pt leaves the earlier best.pt inode unchanged. Distinct
    paths already naming the same inode are rejected, as are identical paths.
    Only known hardlink-unavailable errors trigger a copy fallback. Other link,
    copy, or publication errors propagate while preserving the old destination.
    """
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser()
    # Resolve the parent, retaining the destination entry itself if a symlink
    # happens to exist. Do not accidentally replace its resolved target.
    destination = destination.parent.resolve() / destination.name
    if source == destination or (destination.exists() and os.path.samefile(source, destination)):
        raise ValueError('Checkpoint source and destination must be different files and paths')
    if not source.is_file():
        raise FileNotFoundError('Checkpoint snapshot source must be an existing regular file')
    if destination.is_dir():
        raise IsADirectoryError('Checkpoint snapshot destination must be a file path')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name('.' + destination.name + '.snapshot-' + uuid.uuid4().hex + '.tmp')
    owned_temporary = False
    try:
        try:
            os.link(source, temporary)
            owned_temporary = True
            method = 'hardlink'
        except OSError as error:
            if error.errno not in _LINK_FALLBACK_ERRNOS and getattr(error, 'winerror', None) not in _LINK_FALLBACK_WINERRORS:
                raise
            # Exclusive creation avoids truncating any unexpectedly existing
            # temporary path. copyfile only writes this newly owned file.
            with temporary.open('xb'):
                owned_temporary = True
            shutil.copyfile(source, temporary)
            with temporary.open('r+b') as handle:
                os.fsync(handle.fileno())
            method = 'copy'
        os.replace(temporary, destination)
        owned_temporary = False
        return method
    finally:
        if owned_temporary:
            temporary.unlink(missing_ok=True)
