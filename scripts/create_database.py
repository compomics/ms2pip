#!/usr/bin/env python3
from ms2pip.sqldb import tables

engine = tables.create_engine("postgresql:///ms2pip")
tables.metadata.create_all()
