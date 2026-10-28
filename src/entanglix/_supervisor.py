"""Execute one recorded launch independently of its requesting connection."""

import os
import signal
import sys
from pathlib import Path

from entanglix.installation import hold_use_lock
from entanglix.launches import _arguments, _operation, current_launch, read_launch
from entanglix.runner import _load_record, failure_message, publish_toml


def main():
    path = Path(sys.argv[1])
    descriptor = int(sys.argv[2])
    os.fstat(descriptor)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    signal.signal(signal.SIGHUP, signal.default_int_handler)
    token = current_launch.set(path)
    try:
        hold_use_lock()
        request = _load_record(path / "request.toml")
        _operation(request["operation"])(**_arguments(request["arguments"]))
    except (Exception, KeyboardInterrupt) as exc:
        if read_launch(path)["status"] != "admitted":
            publish_toml(path / "refused.toml", {"reason": failure_message(exc)})
        else:
            print(failure_message(exc), file=sys.stderr)
    finally:
        current_launch.reset(token)
        os.close(descriptor)


if __name__ == "__main__":
    main()
