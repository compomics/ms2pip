import zlib
import numpy
import sqlalchemy

from sqlalchemy import (MetaData, Table, Column, Integer, String, Float,
                        TypeDecorator, Boolean)
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
        value = numpy.asarray(value, dtype=self.dtype)
        return zlib.compress(value.tobytes(), 6)

    def process_result_value(self, value, dialect):
        decompressed = zlib.decompress(value)
        return numpy.frombuffer(decompressed, dtype=self.dtype).tolist()

    def copy(self):
        # NOTE: length will be passed through to BLOB
        return CompressedArray(self.dtype, self.impl.length)


metadata = MetaData()

big_float = numpy.float32.newbyteorder(new_order='B')
big_double = numpy.float64.newbyteorder(new_order='B')

# TODO: index
Entries = Table(
    'entries',
    metadata,
    Column('PrecursorMz', Float, nullable=False),
    Column('PrecursorCharge', Integer, nullable=False),
    Column('PeptideModSeq', String, nullable=False),
    Column('PeptideSeq', String, nullable=False),
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

PeptideToProtein = Table(
    'peptidetoprotein',
    metadata,
    Column('PeptideSeq', String),
    Column('isDecoy', Boolean),
    Column('ProteinAccession', String),
)


def open_sqlite(filename):
    engine = sqlalchemy.create_engine(f"sqlite:///{filename}")
    metadata.bind = engine
    return engine.connect()
