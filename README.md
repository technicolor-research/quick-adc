# Quick ADC #

## Description ##

The Quick ADC project is a C++14 implementation of fast distance computation
techniques for nearest neighbor search in large databases of high-dimensional 
vectors.

Quick ADC builds on Product Quantization (PQ), a widespread nearest neighbor 
search solution. PQ compresses high-dimensional vectors into short codes of a 
few bytes (usually 8-16 bytes). This makes it possible to store very large
databases (billions of vectors) entirely in RAM. To answer nearest neighbor 
queries, PQ relies on Asymmetric Distance Computation (ADC). ADC computes 
distances between the query vector and short codes using lookup tables stored 
in the L1 Cache. ADC offers only limited performance because of the latency of 
the L1 Cache (4-5 cycles), and the lack of parallelism of L1 cache accesses 
(max. 2 concurrent accesses). Our technique, Quick ADC, achieves 4-6x better 
performance than ADC by storing lookup tables in SIMD registers, which offer 
high parallelism and low latency (16 concurrent accesses in 1 cycle).

<p align="center">
    <img src="http://assets.xion345.info/icmr17/quick-adc-overview.png">
</p>

Quick ADC builds on the same key principle as [PQ Fast Scan](https://github.com/technicolor-research/pq-fast-scan).
However, PQ Fast Scan and Quick ADC use different techniques to obtain lookup 
tables that fit SIMD registers. PQ Fast Scan relies on a reorganization of a 
lists of codes, which prevents combining it with inverted indexes, another 
common search acceleration technique. On the contrary, Quick ADC relies on the 
use of 4-bit quantizers instead of common 8-bit quantizers. Quick ADC can be 
combined with inverted indexes. Quick ADC comes at the price of a slight 
decrease in accuracy but this decrease is often negligible.

<p align="center">
    <img src="http://assets.xion345.info/icmr17/qadc-github-graphs.png">
</p>

**Contact:**  
Nicolas Le Souarnec: nicolas.le-scouarnec technicolor.com  
Fabien André: fabien.andre technicolor.com  
Please replace the space by an at sign if you want to send us a mail. You may 
also open a Github issue.

## Publication ##

F. Andre, A.-M. Kermarrec, and N. Le Scouarnec. [Accelerated Nearest Neighbor Search with Quick ADC](https://dl.acm.org/citation.cfm?id=3078992), ICMR, 2017.
Publication presented at ICMR'17: [PDF](https://arxiv.org/abs/1704.07355), [Slides](http://assets.xion345.info/icmr17/slides.pdf), [Poster](http://assets.xion345.info/icmr17/poster.pdf)

## License ##

The Quick ADC project is made available under the Clear BSD license 
terms (See LICENSE file).

Copyright (c) 2017 – Technicolor R&D France

## Building ##

### Requirements ###

Hardware:
* Processor supporting AVX and AVX2 (Intel Haswell or newer)

We have tested Quick ADC on Intel Haswell, Broadwell and Skylake 
processors but we expect it to work on newer generations of Intel processors, as
well as AMD Ryzen processors.

Software:
* Linux distribution from 2016 or newer (Debian 9 or newer, Ubuntu 16.04 or newer etc.)
* [g++](https://gcc.gnu.org/) 5.4 or newer
* [CMake](http://www.cmake.org/) 2.8 or higher
* [OpenCV](http://opencv.org/) Core libraries 
* A sequential (mono-threaded) [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) implementation – **Do not install from packages, see below**
* [Cereal](https://uscilab.github.io/cereal/) 1.2.2 – *See below* 

On Debian-based distributions, you can install g++, opencv core and cmake with the 
following command:

    $ sudo apt-get install build-essential gcc g++ cmake libopencv-core-dev

We recommend creating a working directory for the compilation of this project 
and its dependencies:

    $ mkdir qadc-project
    $ cd qadc-project

### Sequential BLAS implementation ###

Quick ADC requires a high-performance *sequential* BLAS implementation, i.e., a 
BLAS implementation that uses only a *single thread*. Quick ADC already calls BLAS 
functions from multiple threads. Therefore, a BLAS implementation which spawns 
multiple threads will lead to oversubscription, and severely degrade performance. 
The  BLAS implementations included in most Linux distributions are 
multi-threaded, which is why you need to compile a sequential BLAS implementation 
from source.

You can automatically download and compile a *sequential* version of 
OpenBLAS using the `getdeps.sh` script we provide:

    $ sudo apt-get install gfortran wget
    $ git clone https://github.com/technicolor-research/quick-adc
    $ sudo bash quick-adc/getdeps.sh

The `getdeps.sh` script also downloads Cereal, so you may skip the next step if 
you use it. You may use the Intel MKL or ATLAS BLAS implementations instead of
OpenBLAS, but again, make sure to link with the *sequential versions* of these
libraries. 

### Cereal ###

Quick ADC requires the Cereal serialization library. The `getdeps.sh` script
automatically downloads Cereal, so you don't need to do anything more if you 
used it. Otherwise, [download](https://github.com/USCiLab/cereal/archive/v1.2.2.tar.gz) 
and unpack Cereal. Cereal is a header-only library so you don't need to compile 
it. Make sure to download version 1.2.2 because we have found previous versions 
to contain bugs.

### Building Quick ADC ###

If you have not downloaded the project yet:

    $ git clone https://github.com/technicolor-research/quick-adc

You can adjust path and library names in the `quick-adc/LibsPath.cmake` file.
You may need to change `OPENCV_CORE_LIB` from `opencv_core` to `opencv` if your
distribution only provides a full OpenCV implementation (and does not offer a
distinct package for OpenCV Core). You may also need to change 
`SEQUENTIAL_BLAS_LIB` if you don't want to use OpenBLAS.

You may then build the project with the following commands:

    $ mkdir build
    $ cd build
    $ cmake ../quick-adc
    $ make -j4 # Build on 4 cores

## Usage ##

List of executables:

* `db_add` Add vectors to an existing database
* `db_query` Query an existing database (indexed or flat) using the conventional ADC procedure.
* `db_query_4` Query an existing database (indexed of flat) using *Quick ADC*. 
* `flatdb_create` Create a flat database (without an inverted index) of vectors 
* `indexdb_create1` Create an indexed database of vectors (Step 1)
* `indexdb_create2` Create an indexed database of vectors (Step 2)

All executables display a short help message when they are invoked without
arguments:

    $ ./db_add 
    Usage: db_add [db_file] [base_file]

These executables take input files in 
[.fvecs, .ivecs or .bvecs](http://corpus-texmex.irisa.fr/) format. The [SIFT1M](http://corpus-texmex.irisa.fr/), 
[GIST1M](http://corpus-texmex.irisa.fr/), [Deep1M](http://sites.skoltech.ru/compvision/projects/aqtq/) and [SIFT1B](http://corpus-texmex.irisa.fr/) datasets used in our experiments all contain a 
learning set, a base set and a query set in .fvecs or .bvecs format. The ground 
truth is also provided as an .ivecs file.

### Building databases ###

The Quick ADC project allows building flat (without an inverted index) databases 
and indexed databases.

**Building flat datbases**

*(1) Learn a Product Quantizer (PQ), or an Optimized Product Quantizer (OPQ)*

We do not provide scripts or executables to learn product quantizers or 
optimized product quantizers. Our executables expect product quantizers or 
optimized product quantizers to be stored in the file format described in the
[Product quantizer file formats](#product-quantizer-file-formats) section.

You may use the [Quantizations](https://github.com/arbabenko/Quantizations)
project to learn product quantizers or optimized product quantizers. To learn
a product quantizer, you may use the `learnCodebooksPQ` function, as follows:

    learnCodebooksPQ(
        '../sift/sift_learn.fvecs', # learning set
        128, # dimension of vectors
        8,  # number of sub-quantizers
        256, # number of centroids per sub-quantizer 
        # (8-bit quantizer, 256 centroids)
        100000, # number of vectors in the learning set
        'SIFT1M.8x8.pq.pickle', # output file
        iterCount=50 # number of iterations
    )

To learn an optimized product quantizer, you may use the `learnCodebookOPQ`
function, as follows:

    learnCodebooksOPQ(
        '../sift/sift_learn.fvecs',
        100000, # number of vectors in the learning set
        128, # dimensionality of vectors
        8, # number of sub-quantizers
        256, # number of centroids per sub-quantizer
        'SIFT1M.8x8.opq.pickle', # output file
        ninit=50 # number of iterations
    )

The product quantizer and optimized product quantizer files output by these 
functions do not match the file format our executable expect. We provide 
a script to convert to this format:

    $ python convert-quantizer.py pq SIFT1M.8x8.pq.pickle SIFT1M.8x8.pq.data
    $ python convert-quantizer.py opq SIFT1M.8x8.opq.pickle SIFT1M.8x8.opq.data

*(2) Create database*

To create the database `SIFT1M.8x8.opq.flat.db` with the OPQ stored in 
`SIFT1M.8x8.opq.data`:

    $ ./flatdb_create SIFT1M.8x8.opq.data SIFT1M.8x8.opq.flat.db

*(3) Add vectors of the base set to the database*

To add vectors of the base set `../sift/sift_base.fvecs` to the database
`SIFT1M.8x8.opq.flat.db`:

    $ ./db_add SIFT1M.8x8.opq.flat.db ../sift/sift_base.fvecs

**Building indexed databases**

*(1) Build the quantizer for the inverted index (Step 1)*

To create an indexed database with K=256 cells using the learning set 
`../sift/sift_learn.fvecs`:

    $ ./indexdb_create1 256 ../sift/sift_learn.fvecs \
        SIFT1M.256.empty.index.db SIFT1M.256.residuals.fvecs 
        # inverted index with 256 cells

The database is output to `SIFT1M.256.empty.index.db` and the residuals are 
output to `SIFT1M.256.residuals.fvecs`

*(2) Learn a Product Quantizer (PQ), or an Optimized Product Quantizer (OPQ) on*
*the set of residuals*

    learnCodebooksOPQ(
        'SIFT1M.256.residuals.fvecs',
        100000, # number of vectors in the learning set
        128, # dimensionality of vectors
        8, # number of sub-quantizers
        256, # number of centroids per sub-quantizer
        'SIFT1M.256.8x8.opq.pickle', # output file
        ninit=50 # number of iterations
    )

    $ python convert-quantizer.py opq SIFT1M.256.8x8.opq.pickle \
        SIFT1M.256.8x8.opq.data

*(3) Create the database (Step 2)*

To complete the creation of the database `SIFT1M.256.empty.index.db` with the
OPQ `SIFT1M.256.8x8.opq.data` and output the result to `SIFT1M.256.8x8.opq.index.db`:

    $ ./indexdb_create2 SIFT1M.256.empty.index.db SIFT1M.256.8x8.opq.data \
        SIFT1M.256.8x8.opq.index.db

*(4) Add vectors of the base set to the database*

    $ ./db_add SIFT1M.256.8x8.opq.index.db ../sift/sift_base.fvecs

### Querying databases ###

We provide two executables to query databases:
* `db_query` to query databases using the conventional ADC method
* `db_query_4` to query databases using Quick ADC

Both theses executables can be used to query flat databases and indexed databases.

To query the database `SIFT1M.8x8.opq.flat.db` using the query set 
    `../sift/sift_query.fvecs`, and the groundtruth in `../sift/sift_groundtruth.ivecs`:

    $ ./db_query -r100 -b32 SIFT1M.8x8.opq.flat.db ../sift/sift_query.fvecs ../sift/sift_groundtruth.ivecs
    [...]
    r,recall,ma,adc_type,index_us,rotate_us,table_us,scan_us
    100,0.9419,1,adc,0,1,2,2594

Queries are processed by batches of 32 (`-b32`), and the Recall@100 is computed
(`-r100`). The command outputs important metrics as CSV values:

`r`, the r parameter passed on the command line  
`recall`,  the Recall@r  
`ma`, the multiple assignement (always 1 for flat databases, may be selected
for indexed databases)  
`adc_type`, the ADC method used: `adc` for the conventional ADC method, and
`qadc` for Quick ADC.   
`index_us`: time spent in the inverted index (microseconds, always 0 for flat databases)  
`rotate_us`: time spent rotating the vector (microseconds)  
`table_us`: time spent computing distance tables (microseconds)  
`scan_us`: time spent scanning short codes (microseconds)  

When querying indexed databases, the `-m` switch can be used to specify the 
multiple assignement. For instance for a multiple assignement of 24:

    $ ./db_query -r100 -b32 -m24 SIFT1M.256.8x8.opq.index.db ../sift/sift_query.fvecs \ 
        ../sift/sift_groundtruth.ivecs
    [...]
    r,recall,ma,adc_type,index_us,rotate_us,table_us,scan_us
    100,0.9646,24,adc,7,12,46,323

The `db_query_4` executable can only be used with databases using PQ or OPQ with
4-bit quantizers (16 centroids per quantizer). This is because Quick ADC 
requires 4-bit quantizers. For instance, to build an indexed
database compatible with `db_query_4`:

    learnCodebooksOPQ(
        '/home/fabien/postdoc/pq/build-qadc/SIFT1M.256.residuals.fvecs',
        100000,
        128,
        16,
        16,
        'SIFT1M.256.16x4.opq.pickle',
        ninit=50
    )

    $ python convert-quantizer.py opq SIFT1M.256.16x4.opq.pickle SIFT1M.256.16x4.opq.data

    $ ./indexdb_create2 SIFT1M.256.empty.index.db SIFT1M.256.16x4.opq.data \
        SIFT1M.256.16x4.opq.index.db

    $ ./db_add SIFT1M.256.16x4.opq.index.db  ../sift/sift_base.fvecs 

This database can then be queried using `db_query_4`:

    $ ./db_query_4 -r100 -m24 -k0.213 -b32 SIFT1M.256.16x4.opq.index.db ../sift/sift_query.fvecs \
        ../sift/sift_groundtruth.ivecs
    r,recall,ma,adc_type,keep,index_us,rotate_us,table_us,scan_us
    100,0.9426,24,qadc,0.00213,7,13,14,86

The scan time is much lower (86us) when using Quick ADC, than when 
using the conventional ADC method (323us), as shown in the publication.

Compared to `db_query`, `db_query_4` requires an additional parameter: *keep* 
passed via the `-k` switch, which is the percentage of codes scanned NOT using 
SIMD in each partition before using Quick ADC.This parameter is linked to the 
*init* parameter mentionned in the publication, `keep=(init*K)/(ma*N)`, where 
`K` is the number of partitions in the dababase, `ma` the multiple assignement 
and `N`, the totalnumber of codes in the database. In practice, we have found 
values between `keep=0.05%` and `keep=1%` to provide the best accuracy-speed 
ratio.

## Product quantizer file formats ##

Product quantizers or optimized product quantizers should be stored as binary
files, as follows:

**Product Quantizer**

    int32_t dimension; // Dimensionality of vectors
    int32_t m; // Number of sub-quantizers
    int32_t b; // Number of bits per sub-quantizer
    float[m*2^b*dimension/m] codebooks; // Codebooks, stored as a contiguous array

**Optimized Product Quantizer**

    int32_t dimension; // Dimensionality of vectors
    int32_t m; // Number of sub-quantizers
    int32_t b; // Number of bits per sub-quantizer
    float[m*2^b*dimension/m] codebooks; // Codebooks, stored as a contiguous array
    float[dimension*dimension] rotation; // Rotation matrix

## Notes about the source code ##

To process queries, `db_query` and `db_query_4` use an instance of an 
`nns_engine` (or `nns_engine_batch`, the only difference between the two is that
the `nns_engine_batch` processes queries by batches). The `nns_engine` assigns
query vectors to partitions (or cells) of the inverted index, and computes
distance tables. The `nns_engine` then uses a "scanner" to scan the short codes
in the partitions. The class `scanner_simple` (used by `db_query`) is an 
implementation of the conventional ADC method, while `scanner_4` 
(used by `db_query_4`) is an implementation of Quick ADC. The scan functions 
used by `scanner_4` are defined in `scan_simd.hpp`. The current implementation 
of `db_query_4` uses the `scan_avx_4` function (which calls AVX intrinsics). We 
have left an earlier implementation of the scan function named `scan_sse_16_4`, 
which uses only SSE, in the `scan_simd.hpp` file. You may modify `db_query_4.cpp`
to use the older `scan_sse_16_4` instead of `scan_avx_4`.
