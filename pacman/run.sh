#!/bin/bash
for i in {1..200}
do
   python busters.py -l newmap -g RandomGhost -p QLearningAgent -q
   python busters.py -k 4 -l  bigHunt2 -g RandomGhost -p QLearningAgent -q
   python busters.py -k 4 -l  bigHunt3 -g RandomGhost -p QLearningAgent -q
   python busters.py -k 3 -l  bigHunt4 -g RandomGhost -p QLearningAgent -q
   python busters.py -l openHunt -g RandomGhost -p QLearningAgent -q
   python busters.py -k 1 -l labAA1 -g RandomGhost -p QLearningAgent -q
   python busters.py -k 2 -l labAA2 -g RandomGhost -p QLearningAgent -q
   python busters.py -k 3 -l labAA3 -g RandomGhost -p QLearningAgent -q
   python busters.py -k 3 -l labAA4 -g RandomGhost -p QLearningAgent -q
   python busters.py -k 3 -l labAA5 -g RandomGhost -p QLearningAgent -q
done