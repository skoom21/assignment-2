# ------------------------
# ChatGPT-style Python Solver (AC3 + Backtracking)
# ------------------------
import time
from collections import deque

def AC3(domains, neighbors):
    queue = deque([(xi, xj) for xi in domains for xj in neighbors[xi]])
    while queue:
        xi, xj = queue.popleft()
        if revise(domains, xi, xj):
            if not domains[xi]:
                return False
            for xk in neighbors[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True

def revise(domains, xi, xj):
    revised = False
    for x in set(domains[xi]):
        if all(x == y for y in domains[xj]):
            domains[xi].remove(x)
            revised = True
    return revised

def is_valid(grid, r, c, val):
    for i in range(9):
        if grid[r][i] == val or grid[i][c] == val:
            return False
    box_r, box_c = 3 * (r // 3), 3 * (c // 3)
    for i in range(box_r, box_r + 3):
        for j in range(box_c, box_c + 3):
            if grid[i][j] == val:
                return False
    return True

def backtrack(grid):
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                for val in range(1, 10):
                    if is_valid(grid, r, c, val):
                        grid[r][c] = val
                        if backtrack(grid):
                            return True
                        grid[r][c] = 0
                return False
    return True

def solve_chatgpt(grid_str):
    grid = [[0] * 9 for _ in range(9)]
    for i, ch in enumerate(grid_str):
        if ch in '123456789':
            grid[i // 9][i % 9] = int(ch)

    if backtrack(grid):
        return ''.join(str(cell) for row in grid for cell in row)
    return 'No solution'

if __name__ == '__main__':
    with open("input.txt") as f:
        puzzles = [line.strip() for line in f.readlines() if line.strip()]

    start = time.time()
    with open("output_chatgpt.txt", "w") as out:
        for p in puzzles:
            result = solve_chatgpt(p)
            out.write(result + "\n")
    end = time.time()

    print(f"ChatGPT Solver Time: {round((end - start)*1000)} ms")
