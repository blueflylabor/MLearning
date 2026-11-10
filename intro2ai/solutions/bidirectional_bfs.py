from collections import deque

def bidirectional_bfs(capacities, start, goal):
    labels = list(capacities.keys())
    index = {label: i for i, label in enumerate(labels)}
    
    # 1. 正向状态转移：S -> (S_next, action)
    def get_forward_successors(state):
        successors = []
        state_list = list(state)
        
        for b in labels:
            i = index[b]
            # Fill
            if state_list[i] < capacities[b]:
                nxt = list(state_list)
                nxt[i] = capacities[b]
                successors.append((tuple(nxt), f"F{b}"))
            # Empty
            if state_list[i] > 0:
                nxt = list(state_list)
                nxt[i] = 0
                successors.append((tuple(nxt), f"E{b}"))
                
        for src in labels:
            for dst in labels:
                if src == dst: continue
                i, j = index[src], index[dst]
                amt = min(state_list[i], capacities[dst] - state_list[j])
                if amt > 0:
                    nxt = list(state_list)
                    nxt[i] -= amt
                    nxt[j] += amt
                    successors.append((tuple(nxt), f"M{src}{dst}"))
        return successors

    # 2. 反向状态转移：S_curr -> (S_pred, action)
    def get_backward_predecessors(state):
        predecessors = []
        state_list = list(state)
        
        for b in labels:
            i = index[b]
            # Reverse Fill: S_curr[i] 必须为满
            if state_list[i] == capacities[b]:
                for v in range(capacities[b]):
                    pred = list(state_list)
                    pred[i] = v
                    predecessors.append((tuple(pred), f"F{b}"))
            # Reverse Empty: S_curr[i] 必须为 0
            if state_list[i] == 0:
                for v in range(1, capacities[b] + 1):
                    pred = list(state_list)
                    pred[i] = v
                    predecessors.append((tuple(pred), f"E{b}"))
                    
        for src in labels:
            for dst in labels:
                if src == dst: continue
                i, j = index[src], index[dst]
                # Reverse Move: 只有当源为0 或 目的为满 时才可能刚做完 Move
                if state_list[i] == 0 or state_list[j] == capacities[dst]:
                    max_x = min(capacities[src] - state_list[i], state_list[j])
                    for x in range(1, max_x + 1):
                        pred = list(state_list)
                        pred[i] += x
                        pred[j] -= x
                        predecessors.append((tuple(pred), f"M{src}{dst}"))
        return predecessors

    if start == goal:
        return [], [start], 0

    # 双向搜索队列与访问记录
    # forward_visited: state -> (parent_state, action_from_parent)
    # backward_visited: state -> (child_state, action_to_child)
    fwd_queue = deque([start])
    bwd_queue = deque([goal])
    
    fwd_visited = {start: (None, None)}
    bwd_visited = {goal: (None, None)}

    intersect_state = None

    while fwd_queue and bwd_queue:
        # 总是选择前沿节点数较少的一端展开，提高效率
        if len(fwd_queue) <= len(bwd_queue):
            curr = fwd_queue.popleft()
            for nxt, act in get_forward_successors(curr):
                if nxt in bwd_visited:
                    fwd_visited[nxt] = (curr, act)
                    intersect_state = nxt
                    break
                if nxt not in fwd_visited:
                    fwd_visited[nxt] = (curr, act)
                    fwd_queue.append(nxt)
        else:
            curr = bwd_queue.popleft()
            for pred, act in get_backward_predecessors(curr):
                if pred in fwd_visited:
                    bwd_visited[pred] = (curr, act)
                    intersect_state = pred
                    break
                if pred not in bwd_visited:
                    bwd_visited[pred] = (curr, act)
                    bwd_queue.append(pred)

        if intersect_state:
            break

    if not intersect_state:
        return None

    # 重构正向路径
    fwd_actions, fwd_states = [], []
    curr = intersect_state
    while curr != start:
        p_state, act = fwd_visited[curr]
        fwd_actions.append(act)
        fwd_states.append(curr)
        curr = p_state
    fwd_actions.reverse()
    fwd_states.append(start)
    fwd_states.reverse()

    # 重构反向路径
    bwd_actions, bwd_states = [], []
    curr = intersect_state
    while curr != goal:
        c_state, act = bwd_visited[curr]
        bwd_actions.append(act)
        curr = c_state
        bwd_states.append(curr)

    full_actions = fwd_actions + bwd_actions
    full_states = fwd_states + bwd_states

    return full_actions, full_states, len(fwd_visited) + len(bwd_visited)


# 运行测试
problems = [
    ("Part (a)", {"L": 13, "M": 19, "R": 31}, (0, 0, 0), (0, 7, 21)),
    ("Part (b)", {"L": 13, "M": 19, "R": 31}, (5, 13, 19), (0, 7, 21)),
    ("Part (c)", {"L": 13, "R": 31}, (0, 0), (13, 3))
]

for name, caps, start, goal in problems:
    result = bidirectional_bfs(caps, start, goal)
    if result is None:
        print(f"{name}: 未找到解")
        print("-" * 50)
        continue

    actions, states, visited_count = result
    print(f"{name}: 最短步数 = {len(actions)}, 访问状态总数 = {visited_count}")
    print("动作序列:", " -> ".join(actions))
    print("-" * 50)