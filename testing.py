def main():
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

main()