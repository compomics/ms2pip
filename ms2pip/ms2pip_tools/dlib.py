import zlib

import numpy
import sqlalchemy
from sqlalchemy import (Boolean, Column, Float, Index, Integer, MetaData,
                        String, Table, TypeDecorator)
from sqlalchemy.dialects.sqlite import BLOB


DLIB_VERSION = "0.1.14"


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
        value = numpy.asarray(value, dtype=self.dtype)
        return zlib.compress(value.tobytes(), 6)

    def process_result_value(self, value, dialect):
        decompressed = zlib.decompress(value)
        return numpy.frombuffer(decompressed, dtype=self.dtype).tolist()

    def copy(self):
        # NOTE: length will be passed through to BLOB
        return CompressedArray(self.dtype, self.impl.length)


metadata = MetaData()

big_float = numpy.dtype('>f4')
big_double = numpy.dtype('>f8')

Entry = Table(
    'entries',
    metadata,
    Column('PrecursorMz', Float, nullable=False, index=True),
    Column('PrecursorCharge', Integer, nullable=False),
    Column('PeptideModSeq', String, nullable=False),
    Column('PeptideSeq', String, nullable=False, index=True),
    Column('Copies', Integer, nullable=False),
    Column('RTInSeconds', Float, nullable=False),
    Column('Score', Float, nullable=False),
    Column('MassEncodedLength', Integer, nullable=False),
    Column('MassArray', CompressedArray(big_double), nullable=False),
    Column('IntensityEncodedLength', Integer, nullable=False),
    Column('IntensityArray', CompressedArray(big_float), nullable=False),
    Column('CorrelationEncodedLength', Integer, nullable=True),
    Column('CorrelationArray', CompressedArray(big_float), nullable=True),
    Column('RTInSecondsStart', Float, nullable=True),
    Column('RTInSecondsStop', Float, nullable=True),
    Column('MedianChromatogramEncodedLength', Integer, nullable=True),
    Column('MedianChromatogramArray', CompressedArray(big_float), nullable=True),
    Column('SourceFile', String, nullable=False),
)

Index('ix_entries_PeptideModSeq_PrecursorCharge_SourceFile', Entry.c.PeptideModSeq, Entry.c.PrecursorCharge, Entry.c.SourceFile)

PeptideToProtein = Table(
    'peptidetoprotein',
    metadata,
    Column('PeptideSeq', String, nullable=False, index=True),
    Column('isDecoy', Boolean, nullable=True),
    Column('ProteinAccession', String, nullable=False, index=True),
)

Metadata = Table(
    'metadata',
    metadata,
    Column('Key', String, nullable=False, index=True),
    Column('Value', String, nullable=False),
)


def open_sqlite(filename):
    engine = sqlalchemy.create_engine(f"sqlite:///{filename}")
    metadata.bind = engine
    return engine.connect()
