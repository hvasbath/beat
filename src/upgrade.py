"""
Config file upgrading module modified from grond
"""

import sys
import copy
import difflib
from pyrocko import guts_agnostic as aguts
from logging import getLogger


logger = getLogger('upgrade')


def rename_attribute(old, new):
    def func(path, obj):
        if old in obj:
            obj.rename_attribute(old, new)

    return func


def rename_class(new):
    def func(path, obj):
        obj._tagname = new

    return func


def drop_attribute(old):
    def func(path, obj):
        if old in obj:
            obj.drop_attribute(old)

    return func


def set_attribute(k, v):
    def func(path, obj):
        obj[k] = v

    return func


def color_diff(diff):
    green = '\x1b[32m'
    red = '\x1b[31m'
    blue = '\x1b[34m'
    dim = '\x1b[2m'
    reset = '\x1b[0m'

    for line in diff:
        if line.startswith('+'):
            yield green + line + reset
        elif line.startswith('-'):
            yield red + line + reset
        elif line.startswith('^'):
            yield blue + line + reset
        elif line.startswith('@'):
            yield dim + line + reset
        else:
            yield line


def upgrade_config_file(fn, diff=True, update=[]):
    rules = [
        ('beat.SeismicConfig',
            drop_attribute('blacklist')),
        ('beat.WaveformFitConfig',
            set_attribute('blacklist', []))
    ]

    def apply_rules(path, obj):
        for tagname, func in rules:
            if obj._tagname == tagname:
                func(path, obj)

    updates_avail = ['hierarchicals', 'hypers']

    t1 = aguts.load(filename=fn)
    t2 = copy.deepcopy(t1)

    aguts.apply_tree(t2, apply_rules)

    for upd in update:
        if upd not in updates_avail:
            raise TypeError('Update not available for "%s"' % upd)

    if len(update) > 0:
        if 'hypers' in update:
            t2.update_hypers()

        if 'hierarchicals' in update:
            t2.update_hierarchicals()

    s1 = aguts.dump(t1)
    s2 = aguts.dump(t2)

    if diff:
        result = list(difflib.unified_diff(
            s1.splitlines(1), s2.splitlines(1),
            'normalized old', 'normalized new'))

        if sys.stdout.isatty():
            sys.stdout.writelines(color_diff(result))
        else:
            sys.stdout.writelines(result)
    else:
        print(aguts.dump(t2, header=True))
        if 'hypers' in update:
            logger.info(
                'Updated hyper parameters! Previous hyper'
                ' parameter bounds are invalid now!')

        if 'hierarchicals' in update:
            logger.info('Updated hierarchicals.')


if __name__ == '__main__':
    fn = sys.argv[1]
    upgrade_config_file(fn)
