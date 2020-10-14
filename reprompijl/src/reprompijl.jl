module reprompijl
using MPI
using ArgParse
using Printf
using Pkg
using Statistics

@enum Collective begin
    MPI_Allreduce
    MPI_Alltoall
    MPI_Bcast
    MPI_Scan
    Nanosleep
end

struct Args
    nrep::Int64
    calls::Array{Collective,1}
    sizes::Array{Int64,1}
    operation::MPI.Op
    verbose::Bool
    random::Bool
    check::Bool
    summary::Bool
    withOutliers::Bool
end

const root = 0

function str_to_collective(str::SubString{String})::Collective
    if str == "MPI_Allreduce"
        return MPI_Allreduce
    elseif str == "MPI_Alltoall"
        return MPI_Alltoall
    elseif str == "MPI_Bcast"
        return MPI_Bcast
    elseif str == "MPI_Scan"
        return MPI_Scan
    else
        return Nanosleep
    end
end

function parse_parameters()::Args
    s = ArgParseSettings()
    @add_arg_table s begin
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
        "--verbose", "-v"
            help = "increase verbosity level (print times measured for each process)"
            action = :store_true
        "--random", "-r"
            help = "use random data in collective calls (default is all zeros)"
            action = :store_true
        "--check", "-k"
            help = "check the correctness of the calculated result by recalculating on one core"
            action = :store_true
        "--summary"
            help = "only prints statistical data about results"
            action = :store_true
        "--no-remove-outliers"
            help = "don't remove outliers when calculating the mean in --summary mode"
            action = :store_true
    end
    args = parse_args(ARGS, s)
    nrep = args["nrep"]
    calls = [str_to_collective(ss) for ss in  split(args["calls-list"], ",")]
    message_sizes = [parse(Int, ss) for ss in split(args["msizes-list"], ",")]
    verbose = args["verbose"]
    random = args["random"]
    check = args["check"]
    summary = args["summary"]
    withOutliers = args["no-remove-outliers"]
    Args(nrep, calls, message_sizes, MPI.BOR, verbose, random, check, summary, withOutliers)
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
    println("#@verbose=", args.verbose)
    println("#@random=", args.random)
    println("#@check=", args.check)
    println("####")
    println("#@Julia Version=", VERSION)
    println("#@MPI.jl Version=", Pkg.installed()["MPI"])
    println("#@MPI Version=", strip(split(read(`ompi_info`, String), "\n")[2]))
end

function print_verbose(times::Array{Float64,1}, args::Args, call::Collective, msize::Int64)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    # alloc space for times
    if rank == root
        all_times = Array{Float64,2}(undef, args.nrep, size)
    else
        all_times = nothing
    end

    # get times from other processes
    MPI.Gather!(times, all_times, args.nrep, root, comm)

    # print times
    if rank == root
        for proc_id in 1:size
            for i in 1:args.nrep
                Printf.@printf("%7d %50s %10d %12ld %14.10f\n", proc_id,
                    call, i, msize, all_times[i, proc_id])
            end
        end
    end
end

function print_simple(times::Array{Float64,1}, args::Args, call::Collective, size::Int64)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # calculate max runtimes
    maxRuntimes = MPI.Reduce(times, max, root, comm)

    # print results
    if rank == root
        for i in 1:args.nrep
            Printf.@printf("%50s %10d %12ld %14.10f\n", call, i, size, maxRuntimes[i])
        end
    end
end

function print_summary(times::Array{Float64,1}, args::Args, call::Collective, size::Int64)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    # alloc space for times
    if rank == root
        all_times = Array{Float64,2}(undef, args.nrep, size)
    else
        all_times = nothing
    end

    # get times from other processes
    MPI.Gather!(times, all_times, args.nrep, root, comm)

    # calculate max runtime
    maxRuntimes = MPI.Reduce(times, max, root, comm)

    # calculate min runtime
    minRuntimes = MPI.Reduce(times, min, root, comm)

    # print times
    if rank == root
        maxRuntime = maximum(maxRuntimes)

        minRuntime = minimum(minRuntimes)

        # calculate median runtime
        medianRuntime = median(all_times)

        # calculate q1, q3
        q1 = quantile(reshape(all_times, length(all_times)), 0.25)
        q3 = quantile(reshape(all_times, length(all_times)), 0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        if (args.withOutliers)
            meanRuntime = mean(all_times)
        else
            meanRuntime = mean(filter(x -> x >= lower && x <= upper, all_times))
        end

        Printf.@printf("%50s %10d %14.10f %14.10f %14.10f %14.10f\n",
                    call, args.nrep, meanRuntime, medianRuntime, minRuntime, maxRuntime)
    end
end

function print_results(times::Array{Float64,1}, args::Args, call::Collective, size::Int64)
    if args.verbose
        print_verbose(times, args, call, size)
    elseif args.summary
        print_summary(times, args, call, size)
    else
        print_simple(times, args, call, size)
    end
end

function check_allreduce(msize::Int64, send)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    MPI.Barrier(comm)
    
    # alloc space for send buffers
    if rank == root
        all_send = zeros(UInt8, msize, size)
    else
        all_send = nothing
    end

    MPI.Gather!(send, all_send, msize, root, comm)

    result = zeros(UInt8, msize)
    # the only supported operation is MPI.BOR
    if rank == root
        for proc_id in 1:size
            for i in 1:msize
                result[i] = result[i] | all_send[i, proc_id]
            end
        end
    end

    return result
end

function check_scan(msize::Int64, send, recv)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    MPI.Barrier(comm)
    
    # alloc space for send buffers
    if rank == root
        all_send = zeros(UInt8, msize, size)
        all_recv = zeros(UInt8, msize, size)
    else
        all_send = nothing
        all_recv = nothing
    end

    MPI.Gather!(send, all_send, msize, root, comm)
    MPI.Gather!(recv, all_recv, msize, root, comm)

    result = zeros(UInt8, msize, size)
    # the only supported operation is MPI.BOR
    all_correct = true
    if rank == root
        for i in 1:msize
            for proc_id in 1:size
                if proc_id == 1
                    left = all_send[i, 1]
                else
                    left = result[i, proc_id - 1]
                end
                result[i, proc_id] = left | all_send[i, proc_id]
                if result[i, proc_id] != all_recv[i, proc_id]
                    println("Got incorrect result at: [", i, ", ", proc_id, "]")
                    all_correct = false
                end
            end
        end
    end

    return all_correct
end

function bench(args::Args)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    times = zeros(Float64, args.nrep)

    # print header
    if rank == root
        if args.verbose
            Printf.@printf("%7s %50s %10s %12s %14s\n",  "process", "test", "nrep", "count", "time")
        elseif args.summary
            Printf.@printf("%50s %10s %14s %14s %14s %14s\n",
                    "test", "nrep", "mean_sec", "median_sec", "min_sec", "max_sec")
        else
            Printf.@printf("%50s %10s %12s %14s\n", "test", "nrep", "count", "runtime_sec")
        end
    end

    for msize in args.sizes
        for call in args.calls
            # init sync
            MPI.Barrier(comm)

            if call == MPI_Allreduce
                if args.random
                    send = rand(UInt8, msize)
                    recv = rand(UInt8, msize)
                else
                    send = zeros(UInt8, msize)
                    recv = zeros(UInt8, msize)
                end

                if args.check
                    result = check_allreduce(msize, send)
                end

                for i in 1:args.nrep
                    # start sync
                    MPI.Barrier(comm)

                    times[i] = MPI.Wtime()
                    MPI.Allreduce!(send, recv, msize, args.operation, comm)
                    times[i] = MPI.Wtime() - times[i]
                end
            elseif call == MPI_Scan
                if args.random
                    send = rand(UInt8, msize)
                    recv = rand(UInt8, msize)
                else
                    send = zeros(UInt8, msize)
                    recv = zeros(UInt8, msize)
                end

                for i in 1:args.nrep
                    # start sync
                    MPI.Barrier(comm)

                    times[i] = MPI.Wtime()
                    MPI.Scan!(send, recv, msize, args.operation, comm)
                    times[i] = MPI.Wtime() - times[i]
                end
            elseif call == MPI_Alltoall
                if args.random
                    send = rand(UInt8, msize * size)
                    recv = rand(UInt8, msize * size)
                else
                    send = zeros(UInt8, msize * size)
                    recv = zeros(UInt8, msize * size)
                end
                for i in 1:args.nrep
                    # start sync
                    MPI.Barrier(comm)

                    times[i] = MPI.Wtime()
                    MPI.Alltoall!(send, recv, msize, comm)
                    times[i] = MPI.Wtime() - times[i]
                end
            elseif call == MPI_Bcast
                if args.random
                    buf = zeros(UInt8, msize)
                else
                    buf = zeros(UInt8, msize)
                end
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

            all_correct = true
            if rank == root && args.check
                if call == MPI_Allreduce
                    for i in 1:msize
                        if result[i] != recv[i]
                            println("Got an incorrect result for element ", i)
                            all_correct = false
                        end
                    end
                end
            end

            if args.check && call == MPI_Scan
                if !check_scan(msize, send, recv) && rank == root
                    println("Got an incorrect result")
                    all_correct = false
                end
            end

            if rank == root && args.check && all_correct
                println("Verified every element successfully")
            end
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
