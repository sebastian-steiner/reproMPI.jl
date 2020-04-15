module reprompijl
using MPI
using ArgParse
using Printf

@enum Collective begin
    MPI_Allreduce
    MPI_Alltoall
    MPI_Bcast
end

struct Args
    nrep::Int64
    calls::Array{Collective,1}
    sizes::Array{Int64,1}
    operation::MPI.Op
end

function str_to_collective(str::SubString{String})::Collective
    if str == "MPI_Allreduce"
        return MPI_Allreduce
    elseif str == "MPI_Alltoall"
        return MPI_Alltoall
    else
        return MPI_Bcast
    end
end

function parse_parameters()::Args
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--calls-list", "-c"
            help = "list of comma-separated MPI calls to be benchmarked, e.g., --calls-list=MPI_Bcast,MPI_Allgather"
            required = true
            arg_type = String
        "--msizes-list", "-s"
            help = "list of comma-separated message sizes in Bytes, e.g., --msizes-list=10,1024"
            required = true
            arg_type = String
        "--nrep", "-n"
            help = "set number of experiment repetitions"
            arg_type = Int
            default = 10
    end
    args = parse_args(ARGS, s)
    nrep = args["nrep"]
    calls = [str_to_collective(ss) for ss in  split(args["calls-list"], ",")]
    message_sizes = [parse(Int, ss) for ss in split(args["msizes-list"], ",")]
    Args(nrep, calls, message_sizes, MPI.BOR)
end

function print_info(args::Args)
    println("#MPI calls:")
    for c in args.calls
        println("#\t", c)
    end
    println("#Message sizes:")
    for s in args.sizes
        println("#\t", s)
    end
    println("#@operation=", args.operation)
    println("#@datatype=", "UInt8")
    println("#@nrep=", args.nrep)
    println("#@root_proc=", 0)
    println("#@nprocs=", MPI.Comm_size(MPI.COMM_WORLD))
end

function print_results(times::Array{Float64,1}, args::Args, call::Collective, size::Int64)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    root = 0

    # calculate max runtimes
    maxRuntimes = MPI.Reduce(times, max, root, comm)

    # print results
    if rank == root
        for i in 1:args.nrep
            Printf.@printf "%50s %10d %12ld %14.10f\n" call i size maxRuntimes[i]
        end
    end
end

function bench(args::Args)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    times = zeros(Float64, args.nrep)
    root = 0

    Printf.@printf "%50s %10s %12s %14s\n" "test" "nrep" "count" "runtime_sec"

    for msize in args.sizes
        for call in args.calls
            #collective_call() = println("ERROR COLLECTIVE CALL NOT SET")
            #if call == MPI_Allreduce
            #    send = zeros(UInt8, msize)
            #    recv = zeros(UInt8, msize)
            #    collective_call() = MPI.Allreduce!(send, recv, root, args.operation, comm)
            #elseif call == MPI_Alltoall
            #    send = zeros(UInt8, msize * size)
            #    recv = zeros(UInt8, msize * size)
            #    collective_call() = MPI.Alltoall!(send, recv, root, comm)
            #elseif call == MPI_Bcast
            #    buf = zeros(UInt8, msize)
            #    collective_call() = MPI.Bcast!(buf, msize, root, comm)
            #end

            # init sync
            MPI.Barrier(comm)

            if call == MPI_Allreduce
                send = zeros(UInt8, msize)
                recv = zeros(UInt8, msize)
                for i in 1:args.nrep
                    # start sync
                    MPI.Barrier(comm)

                    times[i] = MPI.Wtime()
                    MPI.Allreduce!(send, recv, msize, args.operation, comm)
                    times[i] = MPI.Wtime() - times[i]
                end
            elseif call == MPI_Alltoall
                send = zeros(UInt8, msize * size)
                recv = zeros(UInt8, msize * size)
                for i in 1:args.nrep
                    # start sync
                    MPI.Barrier(comm)

                    times[i] = MPI.Wtime()
                    MPI.Alltoall!(send, recv, msize, comm)
                    times[i] = MPI.Wtime() - times[i]
                end
            elseif call == MPI_Bcast
                buf = zeros(UInt8, msize)
                for i in 1:args.nrep
                    # start sync
                    MPI.Barrier(comm)

                    times[i] = MPI.Wtime()
                    MPI.Bcast!(buf, msize, root, comm)
                    times[i] = MPI.Wtime() - times[i]
                end
            end

            # print timing output
            print_results(times, args, call, msize)
        end
    end
end

function main()
    args = parse_parameters()
    MPI.Init()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    if rank == 0
        print_info(args)
    end
    bench(args)
    MPI.Finalize()
end

main()

end # module
