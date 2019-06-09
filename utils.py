import cmath


def solve_quadratic(a, b, c):
    delta = (b**2) - (4*a*c)
    solution1 = (-b-cmath.sqrt(delta))/(2*a)
    solution2 = (-b+cmath.sqrt(delta))/(2*a)

    return max(solution1.real, solution2.real)


def get_equivalent_compression(input_dim, output_dim, nhu, nhLayers, compression):
    '''
    Attempts to find a suitable hidden layer dimension
    to match the number of parameters in a model compressed
    with HashedNets.
    '''
    if nhLayers == 1 or compression == 1.0:
        return compression

    # Number of compressed parameters for the HashedNet
    # Assumes we hash all biases apart from the output
    N = input_dim * nhu + (nhLayers - 1) * nhu**2 + nhu * output_dim
    biases = nhu + nhu * (nhLayers - 1)
    compressed_N = N * compression + biases * compression + output_dim

    # Solve for compression rate (nhu * compress)
    # inp*nhu*x + nhu*x + layers*nhu*x*nhu*x + layers*nhu*x + oup*nhu*x + oup*x
    # (inp*nhu + nhu + layers*nhu + oup*nhu + oup)*x + layers*nhu*x*nhu*x
    a = (nhLayers - 1) * nhu**2
    b = nhu * (input_dim + 1 + (nhLayers - 1) + output_dim) + output_dim
    c = -compressed_N

    equiv_compression = solve_quadratic(a, b, c)

    c_nhu = nhu * equiv_compression
    equiv_N = (input_dim * c_nhu + c_nhu
               + (nhLayers -1 ) * c_nhu**2
               + (nhLayers - 1) * c_nhu
               + c_nhu * output_dim + output_dim)
    assert abs(equiv_N - compressed_N) < 10, 'Equiv: {} vs. compressed {}'.format(equiv_N, compressed_N)

    return equiv_compression

