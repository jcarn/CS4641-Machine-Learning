Making the Data


You can get my code here: https://github.com/jcarn/CS4641-Machine-Learning


There is a readme in that repository from where I got the code (https://github.com/juanjose49/omscs-cs7641-machine-learning-assignment-4), so if you have any problems refer there, but running is quite simple.


Open the project with eclipse, right click one of the launchers (PokemonLauncher or HardGridWorldLauncher) and click “run as java application”.  If you want to run VI, then uncomment the VI lines in the launcher file, PI, Q is the same. If you want to change epsilon or alpha values, then change the values in the AnalysisRunner or PokeRunner.


Graphing/Tabling


Running the code only prints it in the console, so copy that into an excel file. Use the SPLIT function and pasting as a transpose to make it into a CSV (or just use the csv files I have in the repository). Then, look in plotscores.py in the Graphing directory. At the bottom, just change the name of the csv to whatever file you want to graph.