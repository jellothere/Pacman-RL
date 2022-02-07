from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
from util import manhattanDistance
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
import os
import numpy as np
import random,util,math
from mazeGenerator import *
MAX_DISTANCE = 2147483647

class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalPacmanActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST



class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())

    def getSuccessors(self, currentNode, gameState):
        successors = []
        walls = gameState.getWalls()
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = currentNode
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost =  1
                successors.append( ( nextState, action, cost) )
        return successors

    def aStarSearch(self, position, gameState):
    	# Declaración de lista cerrada (visited) y abierta (bufffer)
        visited = []
        bufffer = util.PriorityQueue()
        initNode = gameState.getPacmanPosition()
        bufffer.push((initNode, []), manhattanDistance(initNode, position))
        # Explorando todos los nodos de la lista abierta
        while not bufffer.isEmpty():  
            currentNode, path = bufffer.pop()
            if (currentNode[0]==position[0] and currentNode[1]==position[1]):
                return path
            # Añadiendo el nodo a la lista abierta
            visited.append(currentNode)
            for succesor, direction, cost in self.getSuccessors(currentNode, gameState):
                if not succesor in visited:
                    newPath = path + [direction]
                    # Heurística empleada - distancia de Manhattan
                    h = cost + manhattanDistance(succesor, position)
                    bufffer.push((succesor, newPath), h) 
        return path

    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)
        index = gameState.data.ghostDistances.index(min(x for x in gameState.data.ghostDistances if x is not None))
        # Obteniendo la posición de los fantasmas
        positions = gameState.getGhostPositions()
        position = positions[index]
        # Comienzo de la búsqueda con A*
        path = self.aStarSearch(position, gameState)
        move = path[0]
        return move

    def printLineData(self, gameState):
    # Modo addition ('a') para que las líneas se añadan una tras otra
        with open('data.txt', 'a') as f:
        # Función write para escribir los datos pedidos
            f.write(str(gameState.getPacmanPosition())
                + ","+ str(gameState.getLegalPacmanActions())
                    +","+ str(gameState.getGhostPositions())
                        + ","+ str(gameState.data.ghostDistances)
                            + ","+ str(gameState.getScore())+ '\n')

class QLearningAgent(BustersAgent):
    """
      Q-Learning Agent
    """
    #Initialization
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self,gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.visited = [gameState.getPacmanPosition()]
        self.actions = []
        self.epsilon = 0
        self.alpha = 0
        self.discount = 0 #para el entrenamiento se uso 0.8
        self.q_table=dict()
  
        #numStates = 2**4
        if os.path.exists("qtable.txt"):
            self.table_file = open("qtable.txt", "r+")
            self.q_table = self.readQtable()    
        else:
            self.table_file = open("qtable.txt", "w+")
            exit(-1)
    
    def readQtable(self):
        "Read qtable from disc"
        return eval(self.table_file.read())
        

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()
        self.table_file.write(str(self.q_table))

    def initialize_Q_values(self, position, legal):
        self.q_table[position] = dict()
        for action in {'North', 'East', 'West', 'South'}:
            if action not in self.q_table[position]:
                self.q_table[position][action] = 0.0
            
    def printQtable(self):
        "Print qtable"
        print(self.q_table)
        print("\n")    


    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()
   

  
    # Obtiene la posicion de la pelota de comida mas cercana entendiendo la posicion de pacman como (0,0)
    def getClosestFoodDot(self, gameState):
        global MAX_DISTANCE
        if not gameState.data.food[0]:
            return None
        pacmanPosition = gameState.getPacmanPosition()
        foodPosition = [-1,-1]
        min = MAX_DISTANCE
        for x, line in enumerate(gameState.data.food):
            for y, dot in enumerate(line):
                if dot:
                    distance = manhattanDistance(pacmanPosition, [x,y])
                    if distance < min:
                        min = distance
                        foodPosition = [x,y]
        return [foodPosition, min]
    
    # Obtiene la posicion de la pelota de comida mas cercana entendiendo la posicion de pacman como (0,0)
    def getClosestFoodDotRelativePosition(self, gameState):
        closestFoodDot = self.getClosestFoodDot(gameState)
        if closestFoodDot[0][0] == -1:
            return [10000,10000]
        x = closestFoodDot[0][0] - gameState.getPacmanPosition()[0]
        y = closestFoodDot[0][1] - gameState.getPacmanPosition()[1]
        return [x,y]

    # Obtiene la posicion del fantasma mas cercano entendiendo la posicion de pacman como (0,0)
    def getClosestGhostRelativePosition(self, gameState):
        ghosts = [x for x in gameState.data.ghostDistances if x is not None]
        if len(ghosts) != 0:
            closest_ghost = gameState.data.ghostDistances.index(min(ghosts))
            x = gameState.getGhostPositions()[closest_ghost][0] - gameState.getPacmanPosition()[0]
            y = gameState.getGhostPositions()[closest_ghost][1] - gameState.getPacmanPosition()[1]
            return [x,y]
        else:
            return [0,0]

    def getRelPositionBoolean(self, relPos):
        res = ["Menor", "Menor"]
        if (relPos[0] > 0):
            res[0] = "Mayor"
        elif relPos[0] == 0:
            res[0] = "Igual"
        if (relPos[1] > 0):
            res[1] = "Mayor"
        elif relPos[1] == 0:
            res[1] = "Igual"
        return res
    
    def getClosestWalls(self, gameState):
        walls = gameState.data.layout.walls
        closest_walls = ()
        pacmanPosition = gameState.getPacmanPosition()

        line = ""
        for i in range(-2,3):
            for j in range(-2,3):
                if (pacmanPosition[1] + i) < 0 or  (pacmanPosition[0] + j) < 0:
                    value = True
                elif (pacmanPosition[1] + i) > (gameState.data.layout.height-1) or  (pacmanPosition[0] + j) > gameState.data.layout.width-1:
                    value = True
                else:
                    value = walls[pacmanPosition[0] + j][pacmanPosition[1] + i]
                closest_walls+=(value,)
                line = line + str(value) + ','
 

        return line, closest_walls

    def getClosestGhostDirection(self, gameState):
        if gameState.data.ghostDistances[0]is not None:
            closest_ghost = gameState.data.ghostDistances.index(min(x for x in gameState.data.ghostDistances if x is not None))
        else:
            return None
        
        ghostX, ghostY = gameState.getGhostPositions()[closest_ghost]
        pacmanX, pacmanY = gameState.getPacmanPosition()

        if(ghostX < pacmanX):
            if(ghostY < pacmanY):
                return "abajo-izq"
            elif(ghostY > pacmanY):
                return "arriba-izq"
            else:
                return "izq"
        elif(ghostX > pacmanX):
            if(ghostY < pacmanY):
                return "abajo-drch"
            elif(ghostY > pacmanY):
                return "arriba-drch"
            else:
                return "drch"
        elif(ghostY > pacmanX):
            return "arriba"
        else:
            return "abajo"

    def getThreeClosestGhosts(self, gameState):
        res = [-1, -1, -1]
        distances = []

        for i in range(0, len(gameState.data.ghostDistances)):
            if(gameState.data.ghostDistances[i] != None):
                distances.append(gameState.data.ghostDistances[i])

        distances_len = len(distances)
        for i in range(0,min(3,distances_len)):
            closest_ghost_index=distances.index(min(x for x in distances if x is not None))
            res[i] = distances.pop(closest_ghost_index)
            distances_len -= distances_len
        
        return res

    def getBooleanActions(self, actions):
        return ['North' in actions,'South' in actions,'East' in actions,'West' in actions]

    def getQuadrants(self, gameState):
        quadrants = [0,0,0,0]
        pacmanX = gameState.getPacmanPosition()[0]
        pacmanY = gameState.getPacmanPosition()[1]
  
        for ghost in gameState.getGhostPositions():
            if (ghost[0] < pacmanX and ghost[1] < pacmanY and ghost[1] > 2):
                quadrants[0] += 1
            if (ghost[0] < pacmanX and ghost[1] >= pacmanY):
                quadrants[1] += 1
            if (ghost[0] >= pacmanX and ghost[1] > 2 and ghost[1] < pacmanY):
                quadrants[2] += 1
            if (ghost[0] >= pacmanX and ghost[1] >= pacmanY):
                quadrants[3] += 1

        return quadrants  

    def printState(self, state):
        strs, walls = self.getClosestWalls(state)
        print(self.getClosestGhostRelativePosition(state)[0])
        print(self.getClosestGhostRelativePosition(state)[1])
        for i in range(0,5):
            for j in range(0,5):
                print(walls[i*5+j], end=' ') 
            print()
        print()

    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        For instance, the state (3,1) is the row 7
        """
        strs, walls = self.getClosestWalls(state)
        booleanActions = self.getBooleanActions(state.getLegalPacmanActions())
        quadrants = self.getQuadrants(state)
        
        position  = (self.getClosestGhostRelativePosition(state)[0],
                    self.getClosestGhostRelativePosition(state)[1])+walls
                        # min(self.getThreeClosestGhosts(state)[0],100),
                        # min(self.getThreeClosestGhosts(state)[1],100),
                        # min(self.getThreeClosestGhosts(state)[2],100),
                        # quadrants[0],
                        # quadrants[1],
                        # quadrants[2],
                        # quadrants[3])+walls
        food = self.getClosestFoodDotRelativePosition(state)
        position += (food[0], food[1])
        return (position)

    def hammingDistance(self, k, target):
            distance = 0
            w = []
            for i in range(2, 27):
                if k[i] != target[i]:  distance+=1
            return distance

    def euclideanDistance(self, k, target):
        return (k[0]-target[0])**2+(k[1]-target[1])**2+(k[27]-target[27])**2+(k[28]-target[28])**2

    def getQValue(self, state, action):

        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        position = self.computePosition(state)
        #initialize Q-value
        if position in self.q_table:
            return self.q_table[position][action]
        else:
            #Computing similar positions already in the table
            similarPosition= min(self.q_table.keys(), key=lambda k: self.euclideanDistance(k, position))
            minimum = self.euclideanDistance(similarPosition, position)
            centroids = [i for i in self.q_table.keys() if self.euclideanDistance(i, position)==minimum]
            if len(centroids) > 1: similarPosition = min(centroids, key=lambda k: self.hammingDistance(k, position))
            return self.q_table[similarPosition][action]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        position = self.computePosition(state)
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions)==0:
          return 0
        if position not in self.q_table:
            self.initialize_Q_values(position, state.getLegalPacmanActions())
      
        return max(self.q_table[self.computePosition(state)][action]for action in legalActions)


    def computeActionFromQValues(self, state, visited):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions)==0:
          return None
        if not visited:
            pos = state.getPacmanPosition()
            legalPositions = dict()
            for action in legalActions:
                if action == 'North':
                    legalPositions[action] = (pos[0],pos[1]+1)
                if action == 'South':
                    legalPositions[action] = (pos[0],pos[1]-1)
                if action == 'East':
                    legalPositions[action] = (pos[0]+1,pos[1])
                if action == 'West':
                    legalPositions[action] = (pos[0]-1,pos[1])
            legalActions = [x for x in legalActions if legalPositions[x] not in self.visited]
       
        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value
        return random.choice(best_actions)
    
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        pos = state.getPacmanPosition()
        visited = False
        # Pick Action
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        action = None

        if len(legalActions) == 0:
             return action
        #Restricting actions to those not repeated
        legalPositions = dict()
        for action in legalActions:
            if action == 'North':
                legalPositions[action] = (pos[0],pos[1]+1)
            if action == 'South':
                legalPositions[action] = (pos[0],pos[1]-1)
            if action == 'East':
                legalPositions[action] = (pos[0]+1,pos[1])
            if action == 'West':
                legalPositions[action] = (pos[0]-1,pos[1])
        flip = util.flipCoin(self.epsilon)
        legalActions = [x for x in legalActions if legalPositions[x] not in self.visited]
        if len(legalActions) == 0:
            #Retracting until available positiions not repeated to be chosen by the policy
            last_action = self.actions.pop()
            if last_action == 'North': return 'South'
            if last_action == 'South': return 'North'
            if last_action == 'East': return 'West'
            if last_action == 'West': return 'East'
        if flip:
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state, visited)
        #Appending the visited positions
        if action == 'North':
            pos = (pos[0],pos[1]+1)
        if action == 'South':
            pos = (pos[0],pos[1]-1)
        if action == 'East':
            pos = (pos[0]+1,pos[1])
        if action == 'West':
            pos = (pos[0]-1,pos[1])
        self.visited.append(pos)
        self.actions.append(action)
        return action

    def update(self, state, action, nextState, reward):
        
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        Good Terminal state -> reward 1
        Bad Terminal state -> reward -1
        Otherwise -> reward 0

        Q-Learning update:

        if terminal_state:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))

        """
        # TRACE for transition and position to update. Comment the following lines if you do not want to see that trace
        #print("Update Q-table with transition: ", state, action, nextState, reward)
        position = self.computePosition(state)
        
        # initialize Q-value
      
        if position not in self.q_table:
            #Blocking further actualization or creation of new states in the table
            return
            # self.initialize_Q_values(position, state.getLegalPacmanActions())
        #print("Corresponding Q-table cell to update:", position, action_column)
        # Funcion de actualizacion no determinista (caso general)
        #	self.alpha: tasa de aprendizaje
        #	self.discount: factor de descuento
        #	reward: refuerzo
        #self.printState(state)
        #print(position)
        self.q_table[position][action] = (1-self.alpha) * self.q_table[position][action] + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))

    def getPolicy(self, state, visited):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state, visited)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)
    
    def getReward(self, state, action, nextstate):
        "Return the obtained reward"
        reward = nextstate.getScore() - state.getScore()
        if reward > 99: self.visited = [nextstate.getPacmanPosition()]
     
        return reward
