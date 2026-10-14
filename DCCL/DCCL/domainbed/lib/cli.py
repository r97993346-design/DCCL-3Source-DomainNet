"""Keep sconf key/value overrides away from argparse's config-file positional."""

import sys


def parse_known_with_overrides(parser, argv=None):
    """Parse normal flags while preserving unknown --key value pairs.

    train_all.py has a positional ``configs`` with nargs='*'. Unprotected
    values of unknown sconf flags can be consumed as config filenames. Attach
    a placeholder to each unknown flag so argparse cannot consume its value,
    then restore sconf's alternating key/value list without interpreting data.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    protected = []
    saved_overrides = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            protected.extend(argv[index:])
            break
        key = token.split("=", 1)[0]
        known = key in parser._option_string_actions
        if token.startswith("--") and parser.allow_abbrev and not known:
            # Let argparse resolve (or reject ambiguous) existing flag
            # abbreviations, including boolean flags without a value.
            known = any(option.startswith(key) for option in parser._option_string_actions)
        if token.startswith("--") and not known:
            if "=" in token:
                key, value = token.split("=", 1)
            else:
                if index + 1 == len(argv) or argv[index + 1].startswith("--"):
                    parser.error("{} requires a value".format(token))
                value = argv[index + 1]
                index += 1
            # A literal space anywhere in an unknown option makes argparse
            # treat that entire token as positional, even with an '='. Keep
            # the original value outside argparse, including spaces and '='.
            token = "{}=__sconf_{}__".format(key, len(saved_overrides))
            saved_overrides[token] = (key, value)
        protected.append(token)
        index += 1

    args, remaining = parser.parse_known_args(protected)
    overrides = []
    for token in remaining:
        if token in saved_overrides:
            overrides.extend(saved_overrides[token])
        else:
            overrides.append(token)
    return args, overrides
