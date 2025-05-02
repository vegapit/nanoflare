#!/usr/bin/env perl
use strict;
use warnings;

my $BASE_DIR = "/Users/korchov/3rdparty";

my $cmd = "cmake -B build";
$cmd = $cmd . " -D CMAKE_BUILD_TYPE=Release";
$cmd = $cmd . " -D FMT_DIR=$BASE_DIR/fmt";
$cmd = $cmd . " -D LIBTORCH_DIR=$BASE_DIR/libtorch";
$cmd = $cmd . " -D NANOBENCH_DIR=$BASE_DIR/nanobench";
$cmd = $cmd . " -D EIGEN_DIR=$BASE_DIR/eigen-3.4.0";
$cmd = $cmd . " .";

print($cmd . "\n");

system( $cmd );