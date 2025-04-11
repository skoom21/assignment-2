# ------------------------
# Python Version (Google OR Tools)
# ------------------------
from ortools.sat.python import cp_model
import time


def solve_sudoku(grid_str):
    model = cp_model.CpModel()
    grid = [[model.NewIntVar(1, 9, f'cell_{r}_{c}') for c in range(9)] for r in range(9)]

    for r in range(9):
        model.AddAllDifferent(grid[r])
    for c in range(9):
        model.AddAllDifferent([grid[r][c] for r in range(9)])
    for br in range(3):
        for bc in range(3):
            block = [grid[r][c] for r in range(br * 3, (br + 1) * 3)
                     for c in range(bc * 3, (bc + 1) * 3)]
            model.AddAllDifferent(block)

    for idx, ch in enumerate(grid_str):
        if ch in '123456789':
            r, c = divmod(idx, 9)
            model.Add(grid[r][c] == int(ch))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0

    class SolutionPrinter(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.result = ''

        def on_solution_callback(self):
            for r in range(9):
                for c in range(9):
                    self.result += str(self.Value(grid[r][c]))

    solution_printer = SolutionPrinter()
    solver.Solve(model, solution_printer)
    return solution_printer.result


if __name__ == '__main__':
    with open("input.txt") as f:
        puzzles = [line.strip() for line in f.readlines() if line.strip()]

    start = time.time()
    with open("output_ortools.txt", "w") as out:
        for p in puzzles:
            result = solve_sudoku(p)
            out.write(result + "\n")
    end = time.time()

    print(f"Google OR Tools Solver Time: {round((end - start)*1000)} ms")