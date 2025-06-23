#!/bin/bash

set -e

cd /home/test/tools/confuzzius

git clone  https://github.com/faustocarva/ConFuzzius 

cd  ConFuzzius
pip3 install cython
pip3 install cytoolz

cd fuzzer && pip3 install -r requirements.txt
