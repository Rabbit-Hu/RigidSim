import warp as wp


def wp_slice(a: wp.array, start, end):
    """Utility function to slice a warp array along the first dimension
    """

    assert a.is_contiguous
    assert 0 <= start <= end <= a.shape[0]
    return wp.array(
        ptr=a.ptr + start * a.strides[0],
        dtype=a.dtype,
        shape=(end - start, *a.shape[1:]),
        strides=a.strides,
        device=a.device,
        copy=False,
        owner=False,
    )
