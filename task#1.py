def find_peak(N: int) -> int:
    def query(x: int) -> int:
        return -1 * (x - 7)**2 + 49

    left, right = 0, N

    while left <= right:
        mid = (left + right) // 2
        current = query(mid)
        left_neighbor = query(mid - 1) if mid - 1 >= left else float('-inf')
        right_neighbor = query(mid + 1) if mid + 1 <= right else float('-inf')

        if current < left_neighbor:
            right = mid - 1
        elif current < right_neighbor:
            left = mid + 1
        else:
            return mid

    return -1

if __name__ == "__main__":
    N = 15
    peak_index = find_peak(N)
    print(f"Peak found at index: {peak_index}")
