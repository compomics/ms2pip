#!/usr/bin/env python3
import argparse
import sys
import zlib
import numpy
import sqlalchemy
import pyteomics.mgf
import ms2pip.config_parser
import ms2pip.peptides

from collections import defaultdict, namedtuple
from sqlalchemy import (MetaData, Table, Column, Integer, String, Float,
                        ForeignKey, TypeDecorator)
from sqlalchemy.sql import select
from sqlalchemy.dialects.sqlite import BLOB


class CompressedArray(TypeDecorator):
    """ Sqlite-like does not support arrays.
        Let's use a custom type decorator.

        See http://docs.sqlalchemy.org/en/latest/core/types.html#sqlalchemy.types.TypeDecorator
    """
    impl = BLOB

    def __init__(self, dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype = dtype

    def process_bind_param(self, value, dialect):
        return zlib.compress(value.tobytes(), 6)

    def process_result_value(self, value, dialect):
        decompressed = zlib.decompress(value)
        return numpy.frombuffer(decompressed, dtype=self.dtype)

    def copy(self):
        return CompressedArray(self.dtype, self.impl.length)


metadata = MetaData()

RefSpectra = Table('RefSpectra', metadata,
                   Column('id', Integer, primary_key=True),
                   Column('peptideSeq', String(200)),
                   Column('precursorMZ', Float),
                   Column('precursorCharge', Integer),
                   Column('retentionTime', Float),
                   )

Modifications = Table('Modifications', metadata,
                      Column('id', Integer, primary_key=True),
                      Column('RefSpectraID', Integer, ForeignKey('RefSpectra.id')),
                      Column('position', Integer),
                      Column('mass', Float),
                      )

RefSpectraPeaks = Table('RefSpectraPeaks', metadata,
                        Column('RefSpectraID', Integer, ForeignKey('RefSpectra.id')),
                        Column('peakMZ', CompressedArray(numpy.double)),
                        Column('peakIntensity', CompressedArray(numpy.single)),
                        )

Modification = namedtuple("Modification", "name, amino_acid")


def open_sqlite(filename):
    engine = sqlalchemy.create_engine(f"sqlite:///{filename}")
    metadata.bind = engine
    return engine.connect()


def get_modification_config(config_file):
    mods = defaultdict(list)

    config = ms2pip.config_parser.ConfigParser(filepath=config_file)
    modifications = ms2pip.peptides.Modifications()
    modifications.add_from_ms2pip_modstrings(config.config['ms2pip']['ptm'])

    for name, mod in modifications._all_modifications:
        mods[mod['mass_shift']].append(Modification(name, mod['amino_acid']))
    return mods


def read_peptides(filename, modifications):
    with open_sqlite(filename) as connection:
        for spec in connection.execute(RefSpectra.select()):
            spec_mods = []
            for modification in connection.execute(Modifications.select().where(Modifications.c.RefSpectraID == spec.id)):
                for mod in modifications[modification.mass]:
                    if mod.amino_acid == 'N-term' and modification.position == 1:
                        spec_mods.append(f"0|{mod.name}")
                    elif mod.amino_acid == 'C-term' and modification.position == len(spec.peptideSeq):
                        spec_mods.append(f"-1|{mod.name}")
                    elif mod.amino_acid == spec.peptideSeq[modification.position-1]:
                        spec_mods.append(f"{modification.position}|{mod.name}")
            spec_mods = '|'.join(spec_mods) if spec_mods else '-'
            yield f"{spec.id} {spec_mods} {spec.peptideSeq} {spec.precursorCharge}"


def read_spectra(filename):
    with open_sqlite(filename) as connection:
        for spec in connection.execute(select([RefSpectra, RefSpectraPeaks]).select_from(RefSpectra.join(RefSpectraPeaks, RefSpectra.c.id == RefSpectraPeaks.c.RefSpectraID))):
            yield {
                'm/z array': spec.peakMZ,
                'intensity array': spec.peakIntensity,
                'params': {
                    'title': spec.id,
                    'pepmass': spec.precursorMZ,
                    'charge': spec.precursorCharge,
                    'rtinseconds': spec.retentionTime
                }
            }


def blib_to_peprec(blib_filename, peprec_file, modifications):
    for peptide in read_peptides(blib_filename, modifications):
        peprec_file.write(f"{peptide}\n")
    peprec_file.flush()


def blib_to_mgf(blib_filename, mgf_file):
    pyteomics.mgf.write(read_spectra(blib_filename), output=mgf_file)


def main():
    parser = argparse.ArgumentParser(description='Convert BiblioSpec Spectral Library to peprec and/or MGF files.')
    parser.add_argument('blib_filename',
                        help='input blib file')
    parser.add_argument('--peprec', nargs='?', type=argparse.FileType('w'),
                        const=sys.stdout,
                        help='write peprec file')
    parser.add_argument('--mgf', nargs='?', type=argparse.FileType('w'),
                        const=sys.stdout,
                        help='write MGF file')
    parser.add_argument('--config',
                        help='MS2PIP config file')

    args = parser.parse_args()

    if args.peprec and not args.config:
        parser.error("--peprec requires --config")

    if args.peprec:
        modifications = get_modification_config(args.config)
        blib_to_peprec(args.blib_filename, args.peprec, modifications)

    if args.mgf:
        blib_to_mgf(args.blib_filename, args.mgf)


if __name__ == "__main__":
    main()
