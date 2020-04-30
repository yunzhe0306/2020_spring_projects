import numpy as np
import copy
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import itertools
import dill
import os


class MC_Player:
    def __init__(self, othello):
        self.othello = othello
        self.forest = {}
        self.file_name = "./forest.pickle"
        self.load_existing_forest()

        if othello is not None:
            self.get_tree(init_board=othello.board)

    def get_tree(self, init_board):
        if init_board.tostring() in self.forest:
            # READ EXISTING TREE
            self.tree = self.forest[init_board.tostring()]
        else:
            # CREATE NEW TREE
            new_tree = MC_Tree(board=init_board)
            self.forest[init_board.tostring()] = new_tree
            self.tree = new_tree

    def load_existing_forest(self):
        if os.path.exists(self.file_name):
            print("Loading MC forest...")
            with open(self.file_name, "rb") as dill_file:
                self.forest = dill.load(dill_file)

    def save_forest(self):
        with open(self.file_name, "wb+") as dill_file:
            dill.dump(self.forest, dill_file)

    def train(self, rounds):
        round_per_phase = 5000
        num_phase = rounds // round_per_phase

        for i in range(num_phase):
            counter = 0
            print("This is phase: ", i, " / ", num_phase)
            while counter <= round_per_phase:
                self.train_MC()
                counter += 1

                if counter % 100 == 0:
                    print("Round: ", counter)
            self.save_forest()

    def train_MC(self):
        # Random Othello game
        othello = Othello(human_chess=1)
        self.get_tree(init_board=othello.board)
        othello.select_human_chess(human_chess=1)
        tree = self.tree

        # TRAIN
        counter = -1
        END_FLAG = False
        while True:
            counter += 1
            if counter % 2 == 0:
                role = 'human'
            else:
                role = 'computer'
            valid_moves, chess_to_modify = othello.get_valid_moves(current_role=role)
            if len(valid_moves) == 0:
                if END_FLAG:
                    break
                else:
                    END_FLAG = True
                    continue

            move = self.predict_next_move(board=np.copy(othello.board), valid_moves=valid_moves)

            father_node = tree.nodes[othello.board.tostring()]
            othello.input_next_move(move_pos=move, role=role, chess_to_modify=chess_to_modify)
            # othello.print_current_board(t=counter)

            # New child node
            if othello.board.tostring() not in tree.nodes:
                self.tree.add_new_node(move=move, new_board=np.copy(othello.board), father_node=father_node)
            elif father_node not in tree.nodes[othello.board.tostring()].father_node_list:
                tree.nodes[othello.board.tostring()].father_node_list.append(father_node)

        # Get game result
        white_win = othello.get_game_result(visualize=False)

        # BACKPROPAGATION
        self.tree.backpropagate(this_board=othello.board, result=white_win)

    def predict_next_move(self, board, valid_moves):
        tree = self.tree
        node_moves = self.tree.nodes[board.tostring()].moves

        # Exploaration
        # TODO: substitute with greedy player when playing with humans
        if len(node_moves) != len(valid_moves):
            e_list = [value for value in valid_moves if value not in node_moves]
            return e_list[random.randint(0, len(e_list)-1)]
        # Exploitation
        else:
            return tree.get_best_move(this_board=board)

    def predict_for_game(self, board, valid_moves, chess_to_modify):
        tree = self.tree
        if board.tostring() not in tree.nodes:
            return chess_to_modify['GREEDY_MOVE']
        else:
            node_moves = self.tree.nodes[board.tostring()].moves
            if len(node_moves) != len(valid_moves):
                return chess_to_modify['GREEDY_MOVE']
            # Exploitation
            else:
                return tree.get_best_move(this_board=board)


class MC_Tree:
    def __init__(self, board):
        self.init_board = board
        self.nodes = {}
        self.nodes[board.tostring()] = Node(board=board, father_node=None)

    def add_new_node(self, move, new_board, father_node):
        # CREATE NEW CHILD
        new_node = Node(board=new_board, father_node=father_node)
        self.nodes[new_board.tostring()] = new_node

        # REGISTER
        father_node.add_child(child_board=new_board, move=move)

    def bp_step(self, node, result, LEVEL_COUNTER):
        if node is not None:
            this_result = result * ((-1) ** LEVEL_COUNTER)
            if this_result == 0:
                score = 0.5
            elif this_result > 0:
                score = 1
            else:
                score = 0
            node.add_result(result=score)
            for father_node in node.father_node_list:
                self.bp_step(father_node, result=result, LEVEL_COUNTER=LEVEL_COUNTER+1)

    def backpropagate(self, this_board, result):
        # --- RESULT SHOULD BE REGARDING THE WHITE CHESS
        # Even and Odd division
        node = self.nodes[this_board.tostring()]
        if node is not None:
            self.bp_step(node=node, result=result, LEVEL_COUNTER=0)

    def get_best_move(self, this_board):
        best_move, highest_score = None, -1
        this_node = self.nodes[this_board.tostring()]
        for child_board in this_node.children:
            child_node = self.nodes[child_board.tostring()]
            score = child_node.get_score() + \
                    (np.sqrt(2) * np.sqrt(2 * np.log(this_node.total_num) / child_node.total_num))
            if score > highest_score:
                best_move = this_node.children_2_moves[child_board.tostring()]
                highest_score = score
        return best_move


class Node:
    def __init__(self, board, father_node):
        self.board = board
        self.moves_2_children = {}
        self.children_2_moves = {}
        self.children = []
        self.moves = []
        self.father_node_list = [father_node]

        self.win_num = 0
        self.total_num = 0

        self.exploration_coef = np.sqrt(2)

    def add_child(self, child_board, move):
        child_board, move = copy.deepcopy(child_board), copy.deepcopy(move)

        self.moves_2_children[tuple(move)] = child_board
        self.children_2_moves[child_board.tostring()] = move
        self.moves.append(move)
        self.children.append(child_board)

    def add_result(self, result):
        # RESULT == 0 OR 1
        self.win_num += result
        self.total_num += 1

    def get_score(self):
        return self.win_num / self.total_num


class Othello:
    def __init__(self, human_chess=1):
        self.size = 8
        # -1: WHITE CHESS, 1: BLACK CHESS, -9: OBSTACLE, 9: CATALYST, 5: Candidate Moves
        self.mappings = {-1: 'o', 1:'x', -9:'S', 9:'C', 0:' ', 5: '-'}
        self.human_chess = None
        self.my_chess = None

        self.black_list = []
        self.white_list = []

        self.directions = [[1, 0], [0, 1], [-1, 0], [0, -1],
                           [1, 1], [-1, -1], [-1, 1], [1, -1]]

        self.board = self.init_board()
        self.add_spicies()
        self.select_human_chess(human_chess=human_chess)

    def select_human_chess(self, human_chess):
        self.human_chess = human_chess
        self.my_chess = -1 * human_chess

    def init_board(self):
        board = np.zeros([self.size, self.size])
        init_pos = self.size // 2
        board[init_pos][init_pos] = board[init_pos-1][init_pos-1] = 1
        self.black_list.append([init_pos, init_pos])
        self.black_list.append([init_pos-1, init_pos-1])

        board[init_pos-1][init_pos] = board[init_pos][init_pos-1] = -1
        self.white_list.append([init_pos-1, init_pos])
        self.white_list.append([init_pos, init_pos-1])

        board.astype(np.int16)

        return board

    def add_spicies(self):
        # Add obstacles and catalysts
        FLAG = True
        while FLAG:
            random_x = random.randint(0, self.size - 1)
            random_y = random.randint(0, self.size - 1)

            if self.board[random_x][random_y] == 0:
                self.board[random_x][random_y] = 9 # Catalyst
                # ADD catalyst to both lists
                self.black_list.append([random_x, random_y])
                self.white_list.append([random_x, random_y])
                FLAG = False

        FLAG = True
        while FLAG:
            random_x = random.randint(0, self.size - 1)
            random_y = random.randint(0, self.size - 1)

            if self.board[random_x][random_y] == 0:
                self.board[random_x][random_y] = -9  # Obstacle
                FLAG = False

    def is_within_board(self, pos):
        size = self.size
        [x, y] = pos
        if x < 0 or x >= size:
            return False
        if y < 0 or y >= size:
            return False
        return True

    def get_valid_moves(self, current_role, this_board=None):
        valid_moves = []
        chess_to_modify = {}
        max_flips, greedy_move = -1, None

        if current_role == 'human':
            chess = self.human_chess
        else:
            chess = self.my_chess

        if chess == 1:
            this_list = self.black_list
        else:
            this_list = self.white_list

        chess_to_modify['CURRENT_ROLE'] = chess

        # ROLE
        for this_pos in this_list:
            for direct in self.directions:
                cur_pos = copy.deepcopy(this_pos)
                OPPOSITE_FLAG = False
                chess_list = []
                cur_pos = list(map(add, cur_pos, direct))
                while self.is_within_board(cur_pos):
                    if chess + self.board[cur_pos[0], cur_pos[1]] == 0:
                        # OPPOSITE CHESS
                        chess_list.append(cur_pos)
                        OPPOSITE_FLAG = True
                        cur_pos = list(map(add, cur_pos, direct))
                        continue
                    elif self.board[cur_pos[0], cur_pos[1]] == 9:
                        # CATALYSTS
                        cur_pos = list(map(add, cur_pos, direct))
                        continue
                    elif self.board[cur_pos[0], cur_pos[1]] == -9:
                        # OBSTACLES
                        break
                    elif self.board[cur_pos[0], cur_pos[1]] == 0:
                        # BLANKS
                        if OPPOSITE_FLAG:
                            # ADD corresponding chess to the dictionary
                            if tuple(cur_pos) in chess_to_modify:
                                for chess_2_change in chess_list:
                                    chess_to_modify[tuple(cur_pos)].append(chess_2_change)
                            else:
                                chess_to_modify[tuple(cur_pos)] = chess_list
                                valid_moves.append(cur_pos)
                            if len(chess_to_modify[tuple(cur_pos)]) > max_flips:
                                max_flips = len(chess_to_modify[tuple(cur_pos)])
                                greedy_move = cur_pos
                        break
                    elif chess == self.board[cur_pos[0], cur_pos[1]]:
                        break

        chess_to_modify['GREEDY_MOVE'] = greedy_move
        return valid_moves, chess_to_modify

    def print_current_board(self, t, alter_board=None, valid_moves=None):
        print("\n\n")
        print("=" * 5 + " Time Step: " + str(t) + " " + "=" * 15)
        print("White chess: o || Black chess: x || C: Catalyst || S: Obstacle")
        print("")
        if alter_board is not None:
            print("\nCandidate Moves (represented by '-'): ")
            print("Valid moves: ", valid_moves, "\n")
            this_board = alter_board
        else:
            this_board = self.board

        d = self.size
        print("   ", end="")
        for y in range(d):
            print(y, end=" ")
        print("")
        for y in range(d):
            print(y, "|", end="")  # print the row #
            for x in range(d):
                piece = this_board[y][x]  # get the piece to print
                print(self.mappings[piece], end=" ")
            print("|")
        print("   ", end="")
        for y in range(d):
            print('-', end=" ")
        print("\n" + "=" * 30)

    def print_candidates(self, t, valid_moves):
        new_board = np.copy(self.board)
        for moves in valid_moves:
            x, y = moves
            new_board[x][y] = 5
        self.print_current_board(t, alter_board=new_board, valid_moves=valid_moves)

    def input_next_move(self, move_pos, role, chess_to_modify):
        if role == 'human':
            chess = self.human_chess
        else:
            chess = self.my_chess

        if chess == 1:
            self.black_list.append(move_pos)
            this_list = self.black_list
            rev_list = self.white_list
        else:
            self.white_list.append(move_pos)
            this_list = self.white_list
            rev_list = self.black_list
        self.board[move_pos[0], move_pos[1]] = chess

        chess_list = chess_to_modify[tuple(move_pos)]
        chess_list.sort()
        chess_list = list(k for k, _ in itertools.groupby(chess_list))

        for chess_2_modify in chess_list:
            # print("Black: ", self.black_list)
            # print("White: ", self.white_list)
            # print("This: ", chess_2_modify)
            # print(move_pos, chess)
            self.board[chess_2_modify[0], chess_2_modify[1]] = chess
            this_list.append(chess_2_modify)
            rev_list.remove(chess_2_modify)

    def random_players(self):
        counter = -1
        END_FLAG = False
        while True:
            counter += 1
            if counter % 2 == 0:
                role = 'human'
            else:
                role = 'computer'
            valid_moves, chess_to_modify = self.get_valid_moves(current_role=role)
            print(valid_moves)
            print(chess_to_modify)
            if len(valid_moves) == 0:
                if END_FLAG:
                    break
                else:
                    END_FLAG = True
                    continue
            move = valid_moves[random.randint(0, len(valid_moves)-1)]
            self.input_next_move(move_pos=move, role=role, chess_to_modify=chess_to_modify)
            self.print_current_board(t=counter)
        white_win = self.get_game_result()
        return white_win

    def get_game_result(self, visualize=True):
        white_count = np.count_nonzero(self.board == -1)
        black_count = np.count_nonzero(self.board == 1)

        if white_count > black_count:
            result = 'RESULT: White chess wins'
            white_win = 1
        elif white_count < black_count:
            result = 'RESULT: Black chess wins'
            white_win = -1
        else:
            result = 'RESULT: Draw'
            white_win = 0

        if visualize:
            print("\n\n")
            print("-"*40)
            print("White chess count: ", white_count)
            print("Black chess count: ", black_count)
            print(result)

        return white_win


def play_game():
    othello = Othello()
    mc_p = MC_Player(othello=othello)

    # Get role of chess
    print("Select your role: White chess (-1) or Black chess (1)?")
    chess_choice = int(input())
    while not (chess_choice == 1 or chess_choice == -1):
        print("Retry!")
        chess_choice = int(input())

    othello.select_human_chess(human_chess=chess_choice)

    time_counter = -1
    divider = 0 if chess_choice == 1 else 1
    END_FLAG = False

    # Begin the game
    while True:
        time_counter += 1
        role = 'human' if time_counter % 2 == divider else 'computer'
        valid_moves, chess_to_modify = othello.get_valid_moves(current_role=role)
        if len(valid_moves) == 0:
            if END_FLAG:
                # END OF THE GAME
                break
            else:
                END_FLAG = True
                continue
        else:
            if END_FLAG:
                END_FLAG = False

        # HUMAN PLAYER ROLE
        if role == 'human':
            othello.print_candidates(t=time_counter, valid_moves=valid_moves)

            # TAKE INPUT
            print("Please input your move as \"x y\"")
            try:
                str_list = input().split()
                user_move = [int(str_list[0]), int(str_list[1])]
                # print("User move", user_move)
            except Exception:
                print("Invalid, try again")
                user_move = [-1, -1]

            while user_move not in valid_moves:
                print("Invalid, try again")
                try:
                    str_list = input().split()
                    user_move = [int(str_list[0]), int(str_list[1])]
                    # print("User move", user_move)
                except Exception:
                    print("Invalid, try again")
                    user_move = [-1, -1]

            othello.input_next_move(move_pos=user_move, role='human', chess_to_modify=chess_to_modify)
            othello.print_current_board(t=time_counter)
        # COMPUTER ROLE
        else:
            move = mc_p.predict_for_game(board=othello.board, valid_moves=valid_moves, chess_to_modify=chess_to_modify)
            othello.input_next_move(move_pos=move, role='computer', chess_to_modify=chess_to_modify)
            othello.print_current_board(t=time_counter)

    # END OF THE GAME
    _ = othello.get_game_result()


if __name__ == '__main__':
    play_game()

    """
    # Training mode
    othello = Othello(human_chess=1)
    mc_p = MC_Player(othello=None)
    mc_p.train(rounds=10000)
    """

    """
    othello = Othello(human_chess=1)
    othello.select_human_chess(human_chess=1)
    othello.print_current_board(t=-1)
    _ = othello.random_players()

    
    othello = Othello(human_chess=1)
    othello.select_human_chess(human_chess=1)
    othello.print_current_board(t=-1)
    othello.random_players()
    
    othello.print_current_board(t=0)
    valid_moves, chess_to_modify = othello.get_valid_moves(current_role='human')
    print(valid_moves)
    print(chess_to_modify)
    othello.input_next_move(move_pos=valid_moves[0], role='human', chess_to_modify=chess_to_modify)
    """
