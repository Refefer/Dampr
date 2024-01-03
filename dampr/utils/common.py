
def filter_by_count(pipe, key_func, filter_func):
    """
    Filters out items by count of appearances
    """

    # Count the terms
    item_count = pipe.map(key_func) \
        .count() \
        .filter(lambda count: filter_func(count[1]))

    return item_count.group_by(lambda x: x[0], lambda x: x[1]) \
            .join(pipe.group_by(key_func)) \
            .reduce(lambda lit, rit: rit, many=True) \
            .map(lambda x: x[1])

