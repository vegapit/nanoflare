#!/usr/bin/env perl
use strict;
use warnings;

use File::Path qw(remove_tree);

my $LIBTORCH_DIR = "/Users/korchov/3rdparty/libtorch";

remove_tree("build");

my $cmd = "cmake -B build";
$cmd = $cmd . " -D CMAKE_BUILD_TYPE=Release";
$cmd = $cmd . " -D LIBTORCH_DIR=$LIBTORCH_DIR";
$cmd = $cmd . " .";

system( $cmd );