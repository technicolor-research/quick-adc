from __future__ import print_function
from math import log
import pickle
import struct
import numpy as np
import sys

def convert_pq(in_filename, out_filename_prefix):
    codebooks = None
    with open(in_filename, 'r') as in_file:
        codebooks = pickle.load(in_file)
    assert codebooks.dtype == np.float32
    m, k, sq_dim = codebooks.shape
    b = log(k, 2)
    dim = sq_dim * m
    out_filename = out_filename_prefix + ".pq.data"
    with open(out_filename, 'wb') as out_file:
        metadata = struct.pack('iii', dim, m, b)
        out_file.write(metadata)
        out_file.write(codebooks.tobytes())

def convert_opq(in_filename, out_filename_prefix):
    codebooks = None
    rotation = None
    with open(in_filename, 'r') as in_file:
        codebooks, rotation = pickle.load(in_file)
    assert codebooks.dtype == np.float32
    assert rotation.dtype == np.float32
    m, k, sq_dim = codebooks.shape
    b = log(k, 2)
    dim = sq_dim * m
    dim1, dim2 = rotation.shape
    assert dim == dim1
    assert dim == dim2
    out_filename = out_filename_prefix + ".opq.data"
    with open(out_filename, 'wb') as out_file:
        metadata = struct.pack('iii', dim, m, b)
        out_file.write(metadata)
        out_file.write(codebooks.tobytes())
        out_file.write(rotation.tobytes())

def usage(progname):
    print("{}: [pq|opq] [in_file] [out_file]".format(progname),
        file=sys.stderr)
    sys.exit(1)

def parse_args(argv):
    if len(argv) < 4:
        usage(argv[0])
    pq_type = argv[1]
    in_file = argv[2]
    out_file = argv[3]
    if not pq_type in ["pq","opq"]:
        print("Invalid type: {}".format(pq_type), file=sys.stderr)
        sys.exit(1)
    suffix = "." + pq_type + ".data"
    if not out_file.endswith(suffix):
        print("Out filename must end with {}.data for type {}".format(
            pq_type, pq_type), file=sys.stderr)
        sys.exit(1)
    out_file_prefix = out_file[:len(out_file)-len(suffix)]
    return (pq_type, in_file, out_file_prefix)

if __name__ == "__main__":
    pq_type, in_file, out_file_prefix = parse_args(sys.argv)
    if pq_type == "pq":
        convert_pq(in_file, out_file_prefix)
    elif pq_type == "opq":
        convert_opq(in_file, out_file_prefix)