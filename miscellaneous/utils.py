# Flatten a list of lists
def flatten_listlist(list_list):
    return [item for sublist in list_list for item in sublist]


def pairwise_loop(iterable):
    return zip(iterable, iterable[1:])


def triple_loop(iterable):
    return zip(iterable, iterable[1:], iterable[2:])


def constrain_range(arr, lo=0, hi=1):
    return arr[(arr >= lo) & (arr <= hi)]


def isin_constrained_range(arr, lo=0, hi=1):
    return (arr >= lo) & (arr <= hi)


def isin_range(df, feature, frange):
    """
    possible use of function
    inrange = np.ones(df.shape[0], dtype=bool)
    for f, r in zip(['X', 'Y', 'Z'], [xlim, ylim, zlim]):
        inrange &= isin_range(df, f, r)
    """
    inrange = (df[feature] > min(frange)) & (df[feature] < max(frange))
    return inrange


def chunks(l, n):
    """Yield successive n-sized chunks from l.
    Used in multiprocessing to split processes up into nb_processes and then chunk the lines
    """
    for i in range(0, len(l), n):
        yield l[i:i + n], range(i, i + len(l[i:i + n]))



