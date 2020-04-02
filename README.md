# 2020_spring_projects
Student final projects are public forks from here.

## Final Project Expectations:

You have considerable flexibility about specifics and you will publish your project openly (as a fork from this project repository) to allow making it part of your portfolio if you choose.  Work alone or in a team of two students. 

Regardless of topic, it must involve notable amounts of original work of your own, though it can of course use existing libraries or be inspired in part by some other published work(s). 

PLAGIARISM IS NOT TOLERATED. From the first commit through all production of documentation and code, it must be crystal clear which, if any, parts of the project were based on or duplicated from any other source(s) all of which must be cited.  This should be so specific that any evaluator can easily see which lines of code are original work and which aren't.  Same for all documentation including images, significant algorithms, etc.

## Example topical ideas from which to choose:

(Making original variations of puzzles and games isn't as difficult as it may seem -- we'll discuss this in class. _Though admittedly, making *good* game variations -- that are well-balanced, strategically interesting, with good replay value_ can take expertise or luck and play-testing with revisions.  I'm not expecting that here, given the short time you have.)

1. Devise your own new _original_ type of logic puzzle or an _original variation_ of existing puzzle type. Your proram should be able to randomly generate many puzzles of your type and to verify that all puzzles generated comply with the standard meta-rule that only one valid solution exists. It needs to output the workable puzzles in a way that a human can print or view them conveniently to try solving them, and support either entering a solution for checking or have the program also display the solution for each puzzle when requested. An interactive UI to also "play" the puzzles after generation is *not* required.

2. OR develop an AI game player for an _original variation_ of some existing strategy game.  If you do this, it needs to be set up so it can either play computer-vs-computer and/or against human players with a reasonable text or graphical UI. 2B. If two teams want to independently develop AI players for the same type of game variant as each other (but using different algorithms, strategies, and/or data structures) so they can compete, that is okay.

3. Computationally 'Solve' a game.  _Background: Some strategic games, especially those of perfect information are known to be "solved". See https://en.wikipedia.org/wiki/Solved_game ._  Sometimes these proofs are done through mathematical analysis, other times through computational verification. If you choose this option, you can either write your own code or modify some existing code that plays a game, to exhaustively analyze a game to attempt to prove if it is "solved" in this way for certain configurations. Slight changes to rules or conditions of a known game can alter this outcome and require reanalysis.

4. Do a detailed written code review with algorithm analysis. In this option, you would not be creating any new original game or puzzle variant, and not even doing significant programming. Instead, you would write a detailed analysis of the algorithmic performance of an existing open source program (related to puzzles or games). Early in your analysis, use a code profiler to identify the functions or sections that significantly impact cpu time performance. Then do your own Big-O/Big-Theta analysis fo those potential "bottlenecks". Include performance measurements from running the program in different situations (such as to verify your Big-O assessment as the data size grows), and profiling results. Try to identify any aspect of the program you can make more efficient. Be aware that even submitting someone else's _analysis_ can easily be plagiarism, so you must be careful. 

## Deliverables and other Requirements:

* Have some fun!
* In your own fork, please replace this README.md file's contents with a good introduction to your own project. 
* Even with options 1-3 above, you will need to describe the overall performance characteristics of your program and explain why you chose the data structures and core algorithm(s) you did. It's in the title of this course!
* If your team has more than one student, take efforts to see that each makes git commits. In addition, your README documentation should include a summary of how you shared the work.
* Recorded video or live in-class presentation of your work. 

