
from collections import deque


def bfs_solution(capacities, start, goal):
    """
    Find a shortest action sequence using BFS.

    capacities: dictionary, e.g. {"L": 13, "M": 19, "R": 31}
    start: initial state tuple
    goal: target state tuple
    """

    # Map basket labels to tuple indices.
    labels = list(capacities.keys())
    index = {label: i for i, label in enumerate(labels)}

    # Generate all possible actions.
    actions = []

    # Fill actions: FL, FM, FR, ...
    actions += ["F" + b for b in labels]

    # Empty actions: EL, EM, ER, ...
    actions += ["E" + b for b in labels]

    # Move actions: MLR, MRM, ...
    actions += [
        "M" + src + dst
        for src in labels
        for dst in labels
        if src != dst
    ]

    # State transition function.
    def apply_action(state, action):
        new_state = list(state)

        if action[0] == "F":
            basket = action[1]
            i = index[basket]
            new_state[i] = capacities[basket]

        elif action[0] == "E":
            basket = action[1]
            i = index[basket]
            new_state[i] = 0

        elif action[0] == "M":
            src = action[1]
            dst = action[2]

            i = index[src]
            j = index[dst]

            # Transfer until source is empty
            # or destination is full.
            moved = min(
                state[i],
                capacities[dst] - state[j]
            )

            new_state[i] -= moved
            new_state[j] += moved

        return tuple(new_state)

    # BFS initialization.
    queue = deque([start])

    # Store the predecessor of every visited state.
    parent = {start: None}

    # Store the action used to reach each state.
    action_to = {}

    # Breadth-first search.
    while queue:

        current = queue.popleft()

        # Goal found.
        if current == goal:
            break

        # Explore all possible actions.
        for action in actions:

            next_state = apply_action(current, action)

            # Ignore actions that do not change the state.
            if next_state == current:
                continue

            # Avoid visiting the same state repeatedly.
            if next_state in parent:
                continue

            # Record predecessor and action.
            parent[next_state] = current
            action_to[next_state] = action

            # Add new state to the queue.
            queue.append(next_state)

    # Check whether a solution exists.
    if goal not in parent:
        return None

    # Reconstruct the shortest path.
    action_path = []
    state_path = [goal]

    current = goal

    while current != start:

        action_path.append(action_to[current])

        current = parent[current]

        state_path.append(current)

    # Reverse the paths.
    action_path.reverse()
    state_path.reverse()

    return action_path, state_path


# ==========================================
# Test all three assignment problems
# ==========================================

problems = [

    # Part (a)
    (
        "Part (a)",
        {"L": 13, "M": 19, "R": 31},
        (0, 0, 0),
        (0, 7, 21)
    ),

    # Part (b)
    (
        "Part (b)",
        {"L": 13, "M": 19, "R": 31},
        (5, 13, 19),
        (0, 7, 21)
    ),

    # Part (c)
    (
        "Part (c)",
        {"L": 13, "R": 31},
        (0, 0),
        (13, 3)
    )
]


for name, capacities, start, goal in problems:

    result = bfs_solution(capacities, start, goal)

    print("\n==============================")
    print(name)
    print("==============================")

    if result is None:
        print("No solution found.")
        continue

    action_path, state_path = result

    print("Initial state:", start)
    print("Goal state:", goal)

    print("Shortest number of actions:",
          len(action_path))

    print("\nAction sequence:")
    print(" -> ".join(action_path))

    print("\nState transitions:")

    for i, action in enumerate(action_path):

        print(
            f"{i + 1:2d}. {action:3s} : "
            f"{state_path[i]} -> {state_path[i + 1]}"
        )
