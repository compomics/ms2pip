import sqlalchemy
from sqlalchemy import Column, Float, ForeignKey, Integer, MetaData, String, Table
from sqlalchemy.types import ARRAY


def create_engine(engine_uri):
    engine = sqlalchemy.create_engine(engine_uri)
    metadata.bind = engine
    return engine


metadata = MetaData()

# TODO: filename + path unique
specfile = Table('specfile', metadata,
                 Column('id', Integer, primary_key=True),
                 Column('filename', String, nullable=False),
                 Column('path', String, nullable=False))

spec = Table('spec', metadata,
             Column('specfile_id', Integer, ForeignKey('specfile.id'), primary_key=True),
             Column('spec_id', String, primary_key=True),
             Column('pepmass', Float, index=True, nullable=False),
             Column('mzs', ARRAY(Float), nullable=False))
