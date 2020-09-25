#!/bin/bash

mpi_version="013"
jl_version="140"
num_tasks="1 32"

cd "${0%/*}/jobs"

for dir in $num_tasks;
do
	cd $dir
	for filename in jl$jl_version-$mpi_version-*.job;
	do
		sbatch $filename
	done
	cd ..
done
