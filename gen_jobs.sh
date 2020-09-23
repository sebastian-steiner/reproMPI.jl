#!/bin/bash

out_dir="../../../repro_results/hydra"

jl_versions="103 140"
mpi_versions="013 014 0143"
#collective_calls="MPI_Bcast MPI_Alltoall MPI_Allreduce"
collective_calls="MPI_Scan"

# write the job file to disk to later sbatch it
# $1    base directory
# $2    version (c, jl140, ...)
# $3    number of nodes
# $4    collective call
# $5    message size
# $6    number of threads
# $7    number of repetitions
create_job_file() {
  filename="$1/$6/$2-$3-$4-$5.job"

  mkdir -p "$1/$6"
  touch $filename

  cat > $filename << EOL
#!/bin/bash
#SBATCH -p q_thesis_steiner
#SBATCH --cpu-freq=High
#SBATCH --time=1:00:00
#SBATCH -N $3
#SBATCH --ntasks-per-node=$6

srun -m block:block \\
  --output "$out_dir/$6/$2-rand-$3-$4-$5.out" \\
  --error "$out_dir/$6/$2-rand-$3-$4-$5.err" \\
  -- julia ../../reprompijl/src/reprompijl.jl \\
  --calls-list=$4 \\
  --msizes-list=$5 \\
  --nrep=$7 \\

EOL
}

#create_job_file "." "c" "4" "MPI_Bcast" "1024" "32" 5000

for jl_version in $jl_versions;
do
  for mpi_version in $mpi_versions;
  do
    for call in $collective_calls;
    do
      for nodes in 4 16 32 36;
      do
        for threads in 1 32;
        do
          # small sizes
          for size in 1 4 16 128 1024 10240;
          do
            create_job_file "jobs" "jl${jl_version}-${mpi_version}" $nodes $call $size $threads 5000
          done
          # medium sizes
          for size in 102400 1024000;
          do
            create_job_file "jobs" "jl${jl_version}-${mpi_version}" $nodes $call $size $threads 1000
          done
        done
      done
    done
  done
done

