"""Resolve all candidate configurations through train_all.main in one process."""

import contextlib
import logging
import sys
from pathlib import Path

from protocol import read_json, require

import train_all


def main():
    cases = read_json(Path(sys.argv[1]))
    original_writer = train_all.get_writer
    original_argv = sys.argv

    try:
        for case in cases:
            writers = []

            def tracked_writer(*args, **kwargs):
                writer = original_writer(*args, **kwargs)
                writers.append(writer)
                return writer

            train_all.get_writer = tracked_writer
            sys.argv = case["argv"]
            with Path(case["log"]).open("w", encoding="utf-8") as stream, \
                    contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                try:
                    train_all.main()
                except SystemExit as exit_status:
                    require(exit_status.code in (None, 0), "argument resolution failed")
                else:
                    raise RuntimeError("parameter preflight did not exit at --show")
                finally:
                    logger = logging.getLogger(train_all.Logger.NAME)
                    for handler in tuple(logger.handlers):
                        logger.removeHandler(handler)
                        handler.close()
                    for writer in writers:
                        if hasattr(writer, "close"):
                            writer.close()
            print(case["log"], flush=True)
    finally:
        sys.argv = original_argv
        train_all.get_writer = original_writer


if __name__ == "__main__":
    main()
