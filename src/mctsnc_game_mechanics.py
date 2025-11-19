from numba import cuda

__version__ = "1.0.1"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"


@cuda.jit(device=True)
def is_action_legal(m, n, board, extra_info, turn, action, legal_actions):
    """Checks whether action defined by index ``action`` is legal and leaves the result (a boolean indicator) in array ``legal_actions`` under that index."""
    # is_action_legal_c4(m, n, board, extra_info, turn, action, legal_actions)
    # is_action_legal_gomoku(m, n, board, extra_info, turn, action, legal_actions)
    is_action_legal_reversi(m, n, board, extra_info, turn, action, legal_actions)


@cuda.jit(device=True)
def take_action(m, n, board, extra_info, turn, action):
    """Takes action defined by index ``action`` during an expansion - modifies the ``board`` and possibly ``extra_info`` arrays."""
    # take_action_c4(m, n, board, extra_info, turn, action)
    # take_action_gomoku(m, n, board, extra_info, turn, action)
    take_action_reversi(m, n, board, extra_info, turn, action)


@cuda.jit(device=True)
def legal_actions_playout(m, n, board, extra_info, turn, legal_actions_with_count):
    """Establishes legal actions and their count during a playout; leaves the results in array ``legal_actions_with_count``."""
    # legal_actions_playout_c4(m, n, board, extra_info, turn, legal_actions_with_count)
    # legal_actions_playout_gomoku(m, n, board, extra_info, turn, legal_actions_with_count)
    legal_actions_playout_reversi(m, n, board, extra_info, turn, legal_actions_with_count)



@cuda.jit(device=True)
def take_action_playout(
    m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count
):
    """Takes action defined by index ``action`` during a playout - modifies the ``board`` and possibly arrays: ``extra_info``, ``legal_actions_with_count``."""
    # take_action_playout_c4(
    #     m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count
    # )
    # take_action_playout_gomoku(m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count)
    take_action_playout_reversi(m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count)

@cuda.jit(device=True)
def compute_outcome(
    m, n, board, extra_info, turn, last_action
):  # any outcome other than {-1, 0, 1} implies status: game ongoing
    """
    Computes and returns the outcome of game state represented by ``board`` and ``extra_info`` arrays.
    Outcomes ``{-1, 1}`` denote a win by minimizing or maximizing player, respectively. ``0`` denotes a tie. Any other outcome denotes an ongoing game.
    """
    # return compute_outcome_c4(m, n, board, extra_info, turn, last_action)
    # return compute_outcome_gomoku(m, n, board, extra_info, turn, last_action)
    return compute_outcome_reversi(m, n, board, extra_info, turn, last_action)


@cuda.jit(device=True)
def is_action_legal_c4(m, n, board, extra_info, turn, action, legal_actions):
    """Functionality of function ``is_action_legal`` for the game of Connect 4."""
    legal_actions[action] = True if extra_info[action] < m else False


@cuda.jit(device=True)
def take_action_c4(m, n, board, extra_info, turn, action):
    """Functionality of function ``take_action`` for the game of Connect 4."""
    extra_info[action] += 1
    row = m - extra_info[action]
    board[row, action] = turn


@cuda.jit(device=True)
def legal_actions_playout_c4(m, n, board, extra_info, turn, legal_actions_with_count):
    """Functionality of function ``legal_actions_playout`` for the game of Connect 4."""
    count = 0
    for j in range(n):
        if extra_info[j] < m:
            legal_actions_with_count[count] = j
            count += 1
    legal_actions_with_count[-1] = count


@cuda.jit(device=True)
def take_action_playout_c4(
    m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count
):
    """Functionality of function ``take_action_playout`` for the game of Connect 4."""
    extra_info[action] += 1
    row = m - extra_info[action]
    board[row, action] = turn


@cuda.jit(device=True)
def compute_outcome_c4(m, n, board, extra_info, turn, last_action):
    """Functionality of function ``compute_outcome`` for the game of Connect 4."""
    last_token = -turn
    j = last_action
    i = m - extra_info[j]
    # N-S
    total = 0
    for k in range(1, 4):
        if i - k < 0 or board[i - k, j] != last_token:
            break
        total += 1
    for k in range(1, 4):
        if i + k >= m or board[i + k, j] != last_token:
            break
        total += 1
    if total >= 3:
        return last_token
    # E-W
    total = 0
    for k in range(1, 4):
        if j + k >= n or board[i, j + k] != last_token:
            break
        total += 1
    for k in range(1, 4):
        if j - k < 0 or board[i, j - k] != last_token:
            break
        total += 1
    if total >= 3:
        return last_token
    # NE-SW
    total = 0
    for k in range(1, 4):
        if i - k < 0 or j + k >= n or board[i - k, j + k] != last_token:
            break
        total += 1
    for k in range(1, 4):
        if i + k >= m or j - k < 0 or board[i + k, j - k] != last_token:
            break
        total += 1
    if total >= 3:
        return last_token
    # NW-SE
    total = 0
    for k in range(1, 4):
        if i - k < 0 or j - k < 0 or board[i - k, j - k] != last_token:
            break
        total += 1
    for k in range(1, 4):
        if i + k >= m or j + k >= n or board[i + k, j + k] != last_token:
            break
        total += 1
    if total >= 3:
        return last_token
    draw = True
    for j in range(n):
        if extra_info[j] < m:
            draw = False
            break
    if draw:
        return 0
    return 2  # anything other than {-1, 0, 1} implies 'game ongoing'


@cuda.jit(device=True)
def is_action_legal_gomoku(m, n, board, extra_info, turn, action, legal_actions):
    """Functionality of function ``is_action_legal`` for the game of Gomoku."""
    i = action // n
    j = action % n
    legal_actions[action] = board[i, j] == 0




@cuda.jit(device=True)
def take_action_gomoku(m, n, board, extra_info, turn, action):
    """Functionality of function ``take_action`` for the game of Gomoku."""
    i = action // n
    j = action % n
    board[i, j] = turn


@cuda.jit(device=True)
def legal_actions_playout_gomoku(
    m, n, board, extra_info, turn, legal_actions_with_count
):
    """Functionality of function ``legal_actions_playout`` for the game of Gomoku."""
    if (
        legal_actions_with_count[-1] == 0
    ):  # time-consuming board scan only if legal actions not established yet
        count = 0
        k = 0
        for i in range(m):
            for j in range(n):
                if board[i, j] == 0:
                    legal_actions_with_count[count] = k
                    count += 1
                k += 1
        legal_actions_with_count[-1] = count


@cuda.jit(device=True)
def take_action_playout_gomoku(
    m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count
):
    """Functionality of function ``take_action_playout`` for the game of Gomoku."""
    i = action // n
    j = action % n
    board[i, j] = turn
    last_legal_action = legal_actions_with_count[legal_actions_with_count[-1] - 1]
    legal_actions_with_count[action_ord] = last_legal_action
    legal_actions_with_count[-1] -= 1


@cuda.jit(device=True)
def compute_outcome_gomoku(m, n, board, extra_info, turn, last_action):
    """Functionality of function ``compute_outcome`` for the game of Gomoku."""
    last_token = -turn
    i = last_action // n
    j = last_action % n
    # N-S
    total = 0
    for k in range(1, 6):
        if i - k < 0 or board[i - k, j] != last_token:
            break
        total += 1
    for k in range(1, 6):
        if i + k >= m or board[i + k, j] != last_token:
            break
        total += 1
    if total == 4:
        return last_token
    # E-W
    total = 0
    for k in range(1, 6):
        if j + k >= n or board[i, j + k] != last_token:
            break
        total += 1
    for k in range(1, 6):
        if j - k < 0 or board[i, j - k] != last_token:
            break
        total += 1
    if total == 4:
        return last_token
    # NE-SW
    total = 0
    for k in range(1, 6):
        if i - k < 0 or j + k >= n or board[i - k, j + k] != last_token:
            break
        total += 1
    for k in range(1, 6):
        if i + k >= m or j - k < 0 or board[i + k, j - k] != last_token:
            break
        total += 1
    if total == 4:
        return last_token
    # NW-SE
    total = 0
    for k in range(1, 6):
        if i - k < 0 or j - k < 0 or board[i - k, j - k] != last_token:
            break
        total += 1
    for k in range(1, 6):
        if i + k >= m or j + k >= n or board[i + k, j + k] != last_token:
            break
        total += 1
    if total == 4:
        return last_token
    draw = True
    for i in range(m):
        for j in range(n):
            if board[i, j] == 0:
                draw = False
                break
    if draw:
        return 0
    return 2  # anything other than {-1, 0, 1} implies 'game ongoing'


@cuda.jit(device=True)
def is_action_legal_reversi(m, n, board, extra_info, turn, action, legal_actions):
    legal_actions[action] = False

    if action == m * n:
        player_has_action = False
        for idx in range(m * n):
            i = idx // n
            j = idx % n

            if board[i, j] != 0:
                continue

            for horizontal in range(-1, 2):
                for vertical in range(-1, 2):
                    if horizontal == 0 and vertical == 0:
                        continue

                    row = i + horizontal
                    col = j + vertical

                    if row < 0 or row >= m or col < 0 or col >= n:
                        continue

                    if board[row, col] == -turn:
                        while True:
                            row += horizontal
                            col += vertical
                            if row < 0 or row >= m or col < 0 or col >= n or board[row, col] == 0:
                                break
                            if board[row, col] == turn:
                                player_has_action = True
                                break

                if player_has_action:
                    break

        if player_has_action:
            return

        opponent = -turn
        opponent_has_action = False
        for idx in range(m * n):
            i = idx // n
            j = idx % n
            if board[i, j] != 0:
                continue
            for horizontal in range(-1, 2):
                for vertical in range(-1, 2):
                    if horizontal == 0 and vertical == 0:
                        continue
                    row = i + horizontal
                    col = j + vertical
                    if row < 0 or row >= m or col < 0 or col >= n:
                        continue
                    if board[row, col] == -opponent:
                        while True:
                            row += horizontal
                            col += vertical
                            if row < 0 or row >= m or col < 0 or col >= n or board[row, col] == 0:
                                break
                            if board[row, col] == opponent:
                                opponent_has_action = True
                                break
                if opponent_has_action:
                    break

        if opponent_has_action:
            legal_actions[action] = True
        return

    i = action // n
    j = action % n

    if board[i, j] != 0:
        return

    for horizontal in range(-1, 2):
        for vertical in range(-1, 2):
            if horizontal == 0 and vertical == 0:
                continue
            row = i + horizontal
            col = j + vertical
            if row < 0 or row >= m or col < 0 or col >= n:
                continue
            if board[row, col] == -turn:
                while True:
                    row += horizontal
                    col += vertical
                    if row < 0 or row >= m or col < 0 or col >= n or board[row, col] == 0:
                        break
                    if board[row, col] == turn:
                        legal_actions[action] = True
                        return


@cuda.jit(device=True)
def take_action_reversi(m, n, board, extra_info, turn, action):
    if action == m * n:
        return

    i = action // n
    j = action % n
    board[i, j] = turn

    for horizontal in range(-1, 2):
        for vertical in range(-1, 2):
            if horizontal == 0 and vertical == 0:
                continue
            row = i + horizontal
            col = j + vertical

            if row < 0 or row >= m or col < 0 or col >= n:
                continue

            if board[row, col] == -turn:
                while True:
                    row += horizontal
                    col += vertical

                    if row < 0 or row >= m or col < 0 or col >= n or board[row, col] == 0:
                        break

                    if board[row, col] == turn:
                        while True:
                            row -= horizontal
                            col -= vertical
                            if row == i and col == j:
                                break
                            board[row, col] = turn
                        break


@cuda.jit(device=True)
def legal_actions_playout_reversi(m, n, board, extra_info, turn, legal_actions_with_count):
    count = 0

    for action_index in range(m * n):
        i = action_index // n
        j = action_index % n

        if board[i, j] != 0:
            continue

        legal = False

        for horizontal in range(-1, 2):
            for vertical in range(-1, 2):
                if horizontal == 0 and vertical == 0:
                    continue

                row = i + horizontal
                col = j + vertical

                if row < 0 or row >= m or col < 0 or col >= n:
                    continue
                
                if board[row, col] == -turn:
                    while True:
                        row += horizontal
                        col += vertical
                        if row < 0 or row >= m or col < 0 or col >= n or board[row, col] == 0:
                            break
                        if board[row, col] == turn:
                            legal = True
                            break
                if legal:
                    break

        if legal:
            legal_actions_with_count[count] = action_index
            count += 1

    if count > 0:
        legal_actions_with_count[-1] = count
        return

    opponent = -turn
    opponent_has_action = False

    for action_index in range(m * n):
        i = action_index // n
        j = action_index % n
        if board[i, j] != 0:
            continue

        legal_opponent = False

        for horizontal in range(-1, 2):
            for vertical in range(-1, 2):
                if horizontal == 0 and vertical == 0:
                    continue
                row = i + horizontal
                col = j + vertical
                if row < 0 or row >= m or col < 0 or col >= n:
                    continue
                if board[row, col] == -opponent:
                    while True:
                        row += horizontal
                        col += vertical
                        if row < 0 or row >= m or col < 0 or col >= n or board[row, col] == 0:
                            break
                        if board[row, col] == opponent:
                            legal_opponent = True
                            break
                if legal_opponent:
                    break

        if legal_opponent:
            opponent_has_action = True
            break

    # aktualny gracz nie ma ruchów, wiec trzeba sprawdzic czy przeciwnik je ma, jeśli nie to zero legal actions == koniec gry
    if opponent_has_action:
        legal_actions_with_count[0] = m * n # m*n = indeks poza planszą = pas
        legal_actions_with_count[-1] = 1
    else:
        legal_actions_with_count[-1] = 0


@cuda.jit(device=True)
def take_action_playout_reversi(m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count):
    if action == m * n: # pas
        return

    i = action // n
    j = action % n
    board[i, j] = turn

    for horizontal in range(-1, 2):
        for vertical in range(-1, 2):
            if horizontal == 0 and vertical == 0:
                continue
            row = i + horizontal
            col = j + vertical
            if row < 0 or row >= m or col < 0 or col >= n:
                continue
            if board[row, col] == -turn:
                while True:
                    row += horizontal
                    col += vertical
                    if row < 0 or row >= m or col < 0 or col >= n or board[row, col] == 0:
                        break
                    if board[row, col] == turn:
                        while True:
                            row -= horizontal
                            col -= vertical
                            if row == i and col == j:
                                break
                            board[row, col] = turn
                        break


@cuda.jit(device=True)
def compute_outcome_reversi(m, n, board, extra_info, turn, last_action):
    player_1 = 0
    player_minus_1 = 0

    for i in range(m):
        for j in range(n):
            if board[i, j] == 1:
                player_1 += 1
            elif board[i, j] == -1:
                player_minus_1 += 1

    has_player_1_action = False
    has_player_minus_1_action = False

    for action_index in range(m * n):
        i = action_index // n
        j = action_index % n

        if board[i, j] != 0:
            continue

        for horizontal in range(-1, 2):
            for vertical in range(-1, 2):
                if horizontal == 0 and vertical == 0:
                    continue

                row = i + horizontal
                col = j + vertical
                row2 = row
                col2 = col

                if 0 <= row < m and 0 <= col < n and board[row, col] == -1:
                    while True:
                        row += horizontal
                        col += vertical
                        if row < 0 or row >= m or col < 0 or col >= n or board[row, col] == 0:
                            break
                        if board[row, col] == 1:
                            has_player_1_action = True
                            break

                if 0 <= row2 < m and 0 <= col2 < n and board[row2, col2] == 1:
                    while True:
                        row2 += horizontal
                        col2 += vertical
                        if row2 < 0 or row2 >= m or col2 < 0 or col2 >= n or board[row2, col2] == 0:
                            break
                        if board[row2, col2] == -1:
                            has_player_minus_1_action = True
                            break

                if has_player_1_action and has_player_minus_1_action:
                    break

        if has_player_1_action and has_player_minus_1_action:
            break

    if has_player_1_action or has_player_minus_1_action:
        return 2

    if player_1 > player_minus_1:
        return 1
    if player_minus_1 > player_1:
        return -1
    return 0








