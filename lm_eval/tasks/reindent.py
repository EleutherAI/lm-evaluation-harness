
"""
Reindent files.
"""

from __future__ import print_function
import sys
import getopt
import codecs
import tempfile
import shutil
import os


def _find_indentation(line, config):
    if len(line) and line[0] in (" ", "\t") and not line.isspace():
        if line[0] == "\t":
            config['is-tabs'] = True
        # Find indentation
        i = 0
        for char in list(line):
            if char not in (" ", "\t"):
                break
            i += 1
        config["from"] = i


def find_indentation(line, config):
    # Find indentation level used in file
    if config['from'] < 0:
        _find_indentation(line, config)

    if config['from'] >= 0:
        # Set old indent
        indent = " " if not config['is-tabs'] else "\t"
        indent = indent * config['from']

        # Set new indent
        newindent = " " if not config['tabs'] else "\t"
        if not config['tabs']:
            newindent = newindent * config['to']

        return indent, newindent

    # Continue to the next line, indentation not found
    return False


def replace_inline_tabs(content, config):
    newcontent = ""
    imagined_i = 0
    for i in range(0, len(content)):
        char = content[i]
        if char == '\t':
            spaces = config['tabsize']-(imagined_i % config['tabsize'])
            newcontent += " " * spaces
            imagined_i += spaces
        else:
            newcontent += char
            imagined_i += 1
    return newcontent


def run(fd_in, fd_out, config):
    while True:
        line = fd_in.readline()
        if not line:
            break
        line = line.rstrip('\r\n')

        # Find indentation style used in file if not set
        if config['from'] < 0:
            indent = find_indentation(line, config)
            if not indent:
                print(line, file=fd_out)
                continue
            indent, newindent = indent

        # Find current indentation level
        level = 0
        while True:
            whitespace = line[:len(indent) * (level + 1)]
            if whitespace == indent * (level + 1):
                level += 1
            else:
                break

        content = line[len(indent) * level:]
        if config['all-tabs']:
            content = replace_inline_tabs(content, config)

        line = (newindent * level) + content
        print(line, file=fd_out)


def run_files(filenames, config):
    for filename in filenames:
        with codecs.open(filename, encoding=config['encoding']) as fd_in:
            if config['dry-run']:
                print("Filename: %s" % filename)
                fd_out = sys.stdout
            else:
                fd_out = tempfile.NamedTemporaryFile(mode='wb', delete=False)
                fd_out.close()
                fd_out = codecs.open(fd_out.name, "wb", encoding=config['encoding'])

            run(fd_in, fd_out, config)

            if not config["dry-run"]:
                fd_out.close()
                shutil.copy(fd_out.name, filename)
                os.remove(fd_out.name)


def main(args):
    config = {
        "dry-run": False,
        "help": False,
        "to": 4,
        "from": -1,
        "tabs": False,
        "encoding": "utf-8",
        "is-tabs": False,
        "tabsize": 4,
        "all-tabs": False
    }
    possible_args = {
        "d":  "dry-run",
        "h":  "help",
        "t:": "to=",
        "f:": "from=",
        "n":  "tabs",
        "e:": "encoding=",
        "s:": "tabsize=",
        "a":  "all-tabs",
    }
    optlist, filenames = getopt.getopt(
        args[1:],
        "".join(possible_args.keys()),
        possible_args.values()
    )

    shortargs, longargs = [], []
    for shortarg in possible_args:
        shortargs.append(shortarg.rstrip(":"))
        longargs.append(possible_args[shortarg].rstrip("="))

    for opt, val in optlist:
        opt = opt.lstrip("-")
        if opt in shortargs:
            opt = longargs[shortargs.index(opt)]
        if isinstance(config[opt], bool):
            config[opt] = True
        elif isinstance(config[opt], int):
            config[opt] = int(val)
        else:
            config[opt] = val

    if config['help']:
        help = """
        Usage: %s [options] filename(s)
        Options:
            -h, --help              Show this message
            -d, --dry-run           Don't save anything, just print
                                    the result
            -t <n>, --to <n>        Convert to this number of spaces
                                    (default: 4)
            -f <n>, --from <n>      Convert from this number of spaces
                                    (default: auto-detect, will also
                                    detect tabs)
            -n, --tabs              Don't convert indentation to spaces,
                                    convert to tabs instead. -t and
                                    --to will have no effect.
            -a, --all-tabs          Also convert tabs used for alignment
                                    in the code (Warning: will replace
                                    all tabs in the file, even if inside
                                    a string)
            -s <n>, --tabsize <n>   Set how many spaces one tab is
                                    (only has an effect on -a, default: 4)
            -e <s>, --encoding <s>  Open files with specified encoding
                                    (default: utf-8)
        """ % args[0]

        # Also removes 8 leading spaces to remove our indentation
        print("\n".join([x[8:] for x in help[1:].split("\n")]))
        sys.exit(0)

    if filenames:
        run_files(filenames, config)
    else:
        run(sys.stdin, sys.stdout, config)
