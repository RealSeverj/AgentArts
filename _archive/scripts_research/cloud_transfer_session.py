"""Upload prepared training inputs over verified SSH; credentials stay in memory."""
from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import time

import paramiko


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--user', default='root')
    p.add_argument('--port', type=int, default=22)
    p.add_argument('--known-hosts', type=Path, default=Path.home() / '.ssh/known_hosts')
    p.add_argument('--put', action='append', nargs=2, metavar=('LOCAL', 'REMOTE'), default=[])
    p.add_argument('--get', action='append', nargs=2, metavar=('REMOTE', 'LOCAL'), default=[])
    args = p.parse_args()
    if not args.put and not args.get:
        p.error('Specify at least one --put or --get')
    client = paramiko.SSHClient()
    client.load_host_keys(str(args.known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = getpass.getpass('SSH password (not saved): ')
    client.connect(args.host, port=args.port, username=args.user, password=password,
                   allow_agent=False, look_for_keys=False, timeout=20, banner_timeout=20, auth_timeout=20)
    password = None
    client.get_transport().set_keepalive(20)
    try:
        with client.open_sftp() as sftp:
            for local, remote in args.put:
                source = Path(local)
                if not source.is_file():
                    raise FileNotFoundError(source)
                try:
                    sftp.stat(remote)
                except FileNotFoundError:
                    pass
                else:
                    raise FileExistsError('Destination already exists; verify it separately instead of overwriting: ' + remote)
                temporary = remote + '.uploading'
                last = time.monotonic()
                start = last
                def progress(sent, total):
                    nonlocal last
                    if time.monotonic() - last >= 20 or sent == total:
                        elapsed = time.monotonic() - start
                        print(json.dumps({'file': source.name, 'sent': sent, 'total': total,
                                          'mib_per_second': round(sent / 1024**2 / max(elapsed, .01), 2)}), flush=True)
                        last = time.monotonic()
                sftp.put(str(source), temporary, callback=progress, confirm=True)
                sftp.chmod(temporary, 0o600)
                sftp.rename(temporary, remote)
                print(json.dumps({'uploaded': source.name, 'remote': remote, 'bytes': source.stat().st_size}), flush=True)
            for remote, local in args.get:
                target = Path(local)
                if target.exists():
                    raise FileExistsError(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.downloading')
                if temporary.exists():
                    raise FileExistsError(temporary)
                # Small report exports use bounded sequential reads; whole-model
                # backups can use OpenSSH's pipelined scp separately.
                expected = sftp.stat(remote).st_size
                with sftp.open(remote, 'rb') as source, temporary.open('xb') as out:
                    while block := source.read(256 * 1024):
                        out.write(block)
                if temporary.stat().st_size != expected:
                    raise ValueError('Downloaded size mismatch: ' + remote)
                temporary.rename(target)
                print(json.dumps({'downloaded': remote, 'local': str(target), 'bytes': expected}), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
