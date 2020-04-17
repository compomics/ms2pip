#!/usr/bin/env python3
import argparse
import os
import pyteomics.mgf
from ms2pip.sqldb import tables
from argparse import ArgumentTypeError as err


class PathType(object):
    def __init__(self, exists=True, type='file', dash_ok=True):
        '''exists:
                True: a path that does exist
                False: a path that does not exist, in a valid parent directory
                None: don't care
           type: file, dir, symlink, None, or a function returning True for valid paths
                None: don't care
           dash_ok: whether to allow "-" as stdin/stdout'''

        assert exists in (True, False, None)
        assert type in ('file', 'dir', 'symlink', None) or hasattr(type, '__call__')

        self._exists = exists
        self._type = type
        self._dash_ok = dash_ok

    def __call__(self, string):
        if string == '-':
            # the special argument "-" means sys.std{in,out}
            if self._type == 'dir':
                raise err('standard input/output (-) not allowed as directory path')
            elif self._type == 'symlink':
                raise err('standard input/output (-) not allowed as symlink path')
            elif not self._dash_ok:
                raise err('standard input/output (-) not allowed')
        else:
            e = os.path.exists(string)
            if self._exists==True:
                if not e:
                    raise err("path does not exist: '%s'" % string)

                if self._type is None:
                    pass
                elif self._type=='file':
                    if not os.path.isfile(string):
                        raise err("path is not a file: '%s'" % string)
                elif self._type=='symlink':
                    if not os.path.symlink(string):
                        raise err("path is not a symlink: '%s'" % string)
                elif self._type=='dir':
                    if not os.path.isdir(string):
                        raise err("path is not a directory: '%s'" % string)
                elif not self._type(string):
                    raise err("path not valid: '%s'" % string)
            else:
                if self._exists==False and e:
                    raise err("path exists: '%s'" % string)

                p = os.path.dirname(os.path.normpath(string)) or '.'
                if not os.path.isdir(p):
                    raise err("parent path is not a directory: '%s'" % p)
                elif not os.path.exists(p):
                    raise err("parent directory does not exist: '%s'" % p)

        return string


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("mgf_file", type=PathType(exists=True, type='file'))
    parser.add_argument("--data-dir",
                        type=PathType(exists=True, type='dir'),
                        default="./data")
    parser.add_argument('--db-uri',
                        default='postgresql:///ms2pip')
    return parser.parse_args()


def add_mgf_to_database(connection, data_dir, mgf_file):
    spec_file = os.path.join(data_dir, mgf_file)
    with connection.begin() as trans:
        specfile = connection.execute(
            tables.specfile.insert().values(
                filename=mgf_file,
                path=data_dir))
        specfile_id = specfile.inserted_primary_key[0]

        with pyteomics.mgf.read(spec_file,
                                use_header=False,
                                convert_arrays=0,
                                read_charges=False) as reader:
            for spectrum in reader:
                if 'pepmass' not in spectrum['params']:
                    continue

                connection.execute(
                    tables.spec.insert().values(
                        specfile_id=specfile_id,
                        spec_id=spectrum['params']['title'],
                        pepmass=spectrum['params']['pepmass'][0],
                        mzs=sorted(spectrum['m/z array'])
                    ))
        trans.commit()


def main():
    args = parse_arguments()
    engine = tables.create_engine(args.db_uri)
    with engine.connect() as connection:
        add_mgf_to_database(connection, os.path.dirname(args.mgf_file), os.path.basename(args.mgf_file))


if __name__ == "__main__":
    main()
