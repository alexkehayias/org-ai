#!/bin/sh

BIN_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
SCRIPT_PATH="$(dirname $BIN_PATH)"

/usr/bin/env -S /bin/sh -c "$BIN_PATH/python3 $SCRIPT_PATH/src/chat.py" "$@"
