#!/bin/bash

set -e -u

# Check if the SQLite file exists
if [ ! -f "$1" ]; then
  echo "SQLite file not found: $1"
  exit 1
fi

# Dump the SQL contents of the SQLite file
sqlite3 "$1" .dump
