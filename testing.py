import bisect
import sys
import io
from collections import defaultdict

def solve():
    import sys
    sys.setrecursionlimit(1 << 25)
    n, k = map(int, sys.stdin.readline().split())
    arr = list(map(int, sys.stdin.readline().split()))
    
    if not arr:
        return
    
    global_max = max(arr)
    
    window_max = []
    current_max = None
    for i in range(len(arr)):
        current_max = max(current_max, arr[i])
        window_max.append(current_max)
    
    single_in_window = []
    for j in range(n):
        has_element_once = False
        for num in arr[j]:
            positions = []
            idx = 0
            while idx < len(arr) and idx <= len(arr[idx]):
                if idx + 1 <= j + k:
                    if arr[idx] == num:
                        positions.append(idx + 1)
                        idx += 1
                else:
                    break
            if idx > j + k:
                continue
            if positions:
                cnt = 0
                for pos in positions:
                    cnt += 1
                    if cnt > 1:
                        break
                if cnt == 1:
                    has_element_once = True
                    break
        single_in_window.append(has_element_once)
    
    new_max = []
    for j in range(n):
        if single_in_window[j]:
            new_val = max(window_max[j], global_max)
            new_max.append(new_val)
        else:
            new_max.append(window_max[j])
    
    result = []
    for val in new_max:
        if val == global_max:
            result.append("Nothing")
        else:
            result.append(str(val))
    
    if result:
        output_line = '\n'.join(result)
        print(output_line)
    else:
        print()

def run_test(input_str: str) -> str:
    """Run solve() on the given input and return its output (no trailing whitespace)."""
    backup_stdin = sys.stdin
    backup_stdout = sys.stdout
    sys.stdin = io.StringIO(input_str)
    sys.stdout = io.StringIO()
    try:
        solve()
        return sys.stdout.getvalue().strip()
    finally:
        sys.stdin = backup_stdin
        sys.stdout = backup_stdout

if __name__ == "__main__":
    tests = [
        {
            "input": "5 3\n1\n2\n2\n3\n3",
            "output": "1\n3\n2"
        },
        {
            "input": "6 4\n3\n3\n3\n4\n4\n2",
            "output": "4\nNothing\n3"
        }
    ]

    for idx, test in enumerate(tests, 1):
        result = run_test(test["input"])
        expected = test["output"].strip()
        status = "PASSED" if result == expected else "FAILED"
        print(f"Test {idx}: {status}")
        if status == "FAILED":
            print("Expected:")
            print(expected)
            print("Got:")
            print(result)