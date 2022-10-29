def check_mpi():
    """
    Check if mpi is available and which rank this thread is
    :returns: if mpi is used, rank of this thread, size of the mpi cluster,
    comm of MPI
    """

    rank = 0
    size = 1
    comm = None
    try:
        # see if we have mpi and/or are upalsing parallel
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() > 1:  # need parallel capabilities
            using_mpi = True
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            comm = MPI.COMM_WORLD
        else:
            using_mpi = False
    except ModuleNotFoundError:
        using_mpi = False

    return using_mpi, rank, size, comm
