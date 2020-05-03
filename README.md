# Othello
Wiki-Link: https://en.wikipedia.org/wiki/Reversi

I am interested in building a gaming client (computer vs. human) for 'Othello'.  

The rules state that the game begins with four disks placed in a square in the middle of the grid, two facing white side up, two pieces with the dark side up, with same-colored disks on a diagonal with each other.

Dark must place a piece with the dark side up on the board, in such a position that there exists at least one straight (horizontal, vertical, or diagonal) occupied line between the new piece and another dark piece, with one or more contiguous light pieces between them.

Players take alternate turns. If one player can not make a valid move, play passes back to the other player. When neither player can move, the game ends. 

## Variants:

(1) The board does not necessarily to be square, and it can be a rectangle.

(2) Some 'Obstacles' exist on the board instead of blanks, and it CANNOT be regarded as neither white chess nor black chess.

(3) Some 'Catalysts' exist on the board instead of blanks, and it CAN be regarded as either white chess or black chess. 

## Algorithms chosen:

MCTS (Monte Carlo Tree Search) and Greedy Algorithm if MC tree is unavailable.

