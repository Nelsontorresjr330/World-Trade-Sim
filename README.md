## World Sim Game 

The full project description is attached onto the repo, this repo is mainly my solution to the presented problem. As a shorter summary for it though, the assignment was to develop a program that was capable of taking in multiple input files that define a world with different countries, their resources & possible transformations/trades. After taking them in, the program uses a heuristic, anytime, depth-first search algorithm to find the best possible trades/transformations (all trades = schedule) and outputs it to the user. The "best" path is defined by trying to maximize a schedule's expected utility. 

So long as the input files setup properly and their file locations in country_scheduler and main are correct, the program should execute fairly easily

To run multiple schedules at once, get to the directory with the program and run `sudo python3 main.py`, the `sudo` will occasionally prompt you for your password, this is necessary for permissions to write the output files.

![Multiple_Schedules](https://github.com/Nelsontorresjr330/World-Trade-Sim/blob/main/Multiple_Schedules.png)

The program should then run, there are 4 initial input questions asked by the program (Max Depth, Frontier Size, Max Multiplier & No. of Schedules)

To run the experimental, faster version, the command is `sudo python3 fast.py`

![Faster](https://github.com/Nelsontorresjr330/World-Trade-Sim/blob/main/Fastest.png)

