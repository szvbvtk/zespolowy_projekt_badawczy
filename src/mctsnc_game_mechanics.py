from numba import cuda

__version__ = "1.0.1"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"


@cuda.jit(device=True)
def is_action_legal(m, n, board, extra_info, turn, action, legal_actions):
    """Checks whether action defined by index ``action`` is legal and leaves the result (a boolean indicator) in array ``legal_actions`` under that index."""
    is_action_legal_reversi(m, n, board, extra_info, turn, action, legal_actions)


@cuda.jit(device=True)
def take_action(m, n, board, extra_info, turn, action):
    """Takes action defined by index ``action`` during an expansion - modifies the ``board`` and possibly ``extra_info`` arrays."""
    take_action_reversi(m, n, board, extra_info, turn, action)


@cuda.jit(device=True)
def legal_actions_playout(m, n, board, extra_info, turn, legal_actions_with_count):
    """Establishes legal actions and their count during a playout; leaves the results in array ``legal_actions_with_count``."""
    legal_actions_playout_reversi(
        m, n, board, extra_info, turn, legal_actions_with_count
    )


@cuda.jit(device=True)
def take_action_playout(
    m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count
):
    take_action_playout_reversi(
        m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count
    )


@cuda.jit(device=True)
def compute_outcome(
    m, n, board, extra_info, turn, last_action
):  # any outcome other than {-1, 0, 1} implies status: game ongoing
    """
    Computes and returns the outcome of game state represented by ``board`` and ``extra_info`` arrays.
    Outcomes ``{-1, 1}`` denote a win by minimizing or maximizing player, respectively. ``0`` denotes a tie. Any other outcome denotes an ongoing game.
    """
    return compute_outcome_reversi(m, n, board, extra_info, turn, last_action)


@cuda.jit(device=True)
def _has_any_move(m, n, board, turn):
    opponent = -turn

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

                if 0 <= row < m and 0 <= col < n and board[row, col] == opponent:
                    while True:
                        row += horizontal
                        col += vertical

                        if row < 0 or row >= m or col < 0 or col >= n:
                            break

                        cell = board[row, col]
                        if cell == 0:
                            break
                        if cell == turn:
                            return True

    return False


@cuda.jit(device=True)
def is_action_legal_reversi(m, n, board, extra_info, turn, action, legal_actions):
    legal_actions[action] = False

    if action == m * n:
        if _has_any_move(m, n, board, turn):
            return

        legal_actions[action] = True
        return

    i = action // n
    j = action % n

    if board[i, j] != 0:
        return

    opponent = -turn

    for horizontal in range(-1, 2):
        for vertical in range(-1, 2):
            if horizontal == 0 and vertical == 0:
                continue

            row, col = i + horizontal, j + vertical

            if 0 <= row < m and 0 <= col < n and board[row, col] == opponent:
                while True:
                    row += horizontal
                    col += vertical
                    if not (0 <= row < m and 0 <= col < n):
                        break

                    cell = board[row, col]
                    if cell == 0:
                        break
                    if cell == turn:
                        legal_actions[action] = True
                        return


@cuda.jit(device=True)
def take_action_reversi(m, n, board, extra_info, turn, action):
    if action == m * n:
        return

    i = action // n
    j = action % n

    board[i, j] = turn

    if turn == -1:
        extra_info[0] += 1
    else:
        extra_info[1] += 1

    opponent = -turn

    for horizontal in range(-1, 2):
        for vertical in range(-1, 2):
            if horizontal == 0 and vertical == 0:
                continue

            row, col = i + horizontal, j + vertical

            found_anchor = False
            r_check, c_check = row, col

            if (
                0 <= r_check < m
                and 0 <= c_check < n
                and board[r_check, c_check] == opponent
            ):
                while True:
                    r_check += horizontal
                    c_check += vertical
                    if not (0 <= r_check < m and 0 <= c_check < n):
                        break

                    cell = board[r_check, c_check]
                    if cell == 0:
                        break
                    if cell == turn:
                        found_anchor = True
                        break

            if found_anchor:
                curr_r, curr_c = row, col
                while curr_r != r_check or curr_c != c_check:
                    board[curr_r, curr_c] = turn

                    if turn == -1:
                        extra_info[0] += 1
                        extra_info[1] -= 1
                    else:
                        extra_info[1] += 1
                        extra_info[0] -= 1

                    curr_r += horizontal
                    curr_c += vertical


@cuda.jit(device=True)
def legal_actions_playout_reversi(
    m, n, board, extra_info, turn, legal_actions_with_count
):
    count = 0
    opponent = -turn

    for action in range(m * n):
        i = action // n
        j = action % n

        if board[i, j] != 0:
            continue

        is_legal = False
        for horizontal in range(-1, 2):
            for vertical in range(-1, 2):
                if horizontal == 0 and vertical == 0:
                    continue

                row, col = i + horizontal, j + vertical
                if 0 <= row < m and 0 <= col < n and board[row, col] == opponent:
                    while True:
                        row += horizontal
                        col += vertical
                        if not (0 <= row < m and 0 <= col < n):
                            break
                        cell = board[row, col]
                        if cell == 0:
                            break
                        if cell == turn:
                            is_legal = True
                            break
                if is_legal:
                    break
            if is_legal:
                break

        if is_legal:
            legal_actions_with_count[count] = action
            count += 1

    if count > 0:
        legal_actions_with_count[-1] = count
        return

    if _has_any_move(m, n, board, opponent):
        legal_actions_with_count[0] = m * n
        legal_actions_with_count[-1] = 1
    else:
        legal_actions_with_count[-1] = 0


@cuda.jit(device=True)
def take_action_playout_reversi(
    m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count
):
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
                    if (
                        row < 0
                        or row >= m
                        or col < 0
                        or col >= n
                        or board[row, col] == 0
                    ):
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
    # OPTYMALIZACJA:
    # 1. Sprawdzamy czy gra trwa.
    #    Gra trwa jeśli:
    #    A. Aktualny gracz ma ruchy.
    #    B. LUB poprzedni ruch to NIE BYŁ pas, a aktualny gracz musi spasować (ale przeciwnik może będzie miał ruch).

    # Szybki check: czy aktualny gracz ma ruch?
    if _has_any_move(m, n, board, turn):
        return 2  # Gra trwa

    # Jeśli aktualny gracz nie ma ruchu, to albo PAS, albo KONIEC.
    # Sprawdzamy czy ostatni ruch był pasem. Jeśli tak i teraz też nie ma ruchu -> dwa pasy -> koniec.
    if last_action == m * n:
        # Dwa pasy z rzędu (lub pas i brak możliwości ruchu) -> Koniec gry
        pass
    else:
        # Ostatni ruch to był normalny ruch. Sprawdźmy czy przeciwnik (który teraz będzie miał ruch po naszym pasie) ma opcje.
        if _has_any_move(m, n, board, -turn):
            return 2  # Gra trwa (będzie pas)

    # JEŚLI DOTARLIŚMY TU -> KONIEC GRY
    # Pobieramy wynik bezpośrednio z extra_info (czas O(1) zamiast skanowania O(N))
    # extra_info[0] = BIALE (-1), extra_info[1] = CZARNE (1)

    n_white = extra_info[0]
    n_black = extra_info[1]

    if n_black > n_white:
        return 1
    elif n_white > n_black:
        return -1
    else:
        return 0
