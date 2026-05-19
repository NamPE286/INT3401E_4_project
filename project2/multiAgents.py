# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The evaluation function considers four values: 
        + distanceToFoodValue - distance between Pacman and food
        + ghostScaredTimeValue - the amount of time all the ghosts are in scared state.
        + ghostDistanceValue - distance between Pacman and all the food
        + foodRewardValue - consuming food will be prioritised

        Evalutation function is designed so that Pacman prioritises maximizing score by eating as much food as possible.
        Though if the distance with ghosts are sufficiently close, Pacman will prioritise avoiding instead.
        Pacman is also designed to never stop moving to avoid wasting point by existing.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        # Return Pacman Position in (x, y) form
        newFood = successorGameState.getFood()
        # Return a grid of food's coordinates, e.g. [(x1, y1), (x2, y2),...]
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
         # Return a list of ghost's scared timer
        "*** YOUR CODE HERE ***"
        import numpy as np
        distanceToFoodValue = 0
        ghostScaredTimeValue = 0
        foodRewardValue= 0
        ghostDistanceValue = 0
        finalValue = 0

        distance = 10000
        rightNextToGhost = False
        for food in newFood.asList():
            newDistance = manhattanDistance(food, newPos)
            if distance > newDistance:
                distance = newDistance
        distanceToFoodValue = 1 / (distance) # The closer pacman is to the closest food, the higher the value

        ghostScaredTimeValue = 0
        for time in newScaredTimes:
            ghostScaredTimeValue += time # the more time the ghosts are in scared state, the higher the score
        
        ghostDistanceValue = 0
        for ghost in newGhostStates:
            manhDistance = manhattanDistance(ghost.getPosition(), newPos)
            if ghost.scaredTimer == 0:  # ghost is dangerous
                if manhDistance <= 2:
                    rightNextToGhost = True
                ghostDistanceValue -= 1 / (manhDistance + 1)  # If ghost is dangerous, the value goes down
            else:  # ghost is scared
                ghostDistanceValue += 1 / (manhDistance + 1)    # If ghost is not dangerous, can also prioritise eating ghost.

        # Reward for eating food, arose from the problem of Pacman constantly moving between 2 food items without ever committing to eat
        # either of them
        foodLeft = successorGameState.getFood().count()
        foodEaten = currentGameState.getFood().count() - foodLeft
        foodRewardValue = 5000*foodEaten  # big bonus for actually consuming food

        if rightNextToGhost:
            finalValue = 2500*distanceToFoodValue + 10000*ghostScaredTimeValue + 500000*ghostDistanceValue + foodRewardValue 
            # prioritise ghost distancing if they're close
        else: 
            # distance to food weighs more than distance to ghosts otherwise
            finalValue = 2500*distanceToFoodValue + 10000*ghostScaredTimeValue + 500*ghostDistanceValue + foodRewardValue 

        if (action == Directions.STOP):
            finalValue -= 1000
        return finalValue

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)       

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            succState = gameState.generateSuccessor(0, action)
            value = self.miniMax(succState, 1, 0)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
    
    def miniMax(self, gameState: GameState, agentIndex: int, depth: int):
        """
        A specialized minimax algorithm for the project. getAction() will call this method for every successor states of Pacman agent,
        and the method will return the maximized value for each of them.

        Method is designed to also be usable with ghost agents (agentIndex != 0, which will minimizes the state's value instead).
        """
        if (gameState.isWin() or gameState.isLose()) or depth == self.depth:
            return scoreEvaluationFunction(gameState)
        if agentIndex == 0: # Agent is Pacman
            return self.maxPlaying(gameState, agentIndex, depth)
        else: # Agent is a ghost
            return self.minPlaying(gameState, agentIndex, depth)

    def maxPlaying(self, gameState: GameState, agentIndex: int, depth: int):
        """
        Helper method for miniMax() that returns the value of a state that maximizes. It will go through all of its successors
        and choose the maximum value as its own value.
        """
        value = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            succState = gameState.generateSuccessor(agentIndex, action)
            nextAgent = agentIndex + 1
            if nextAgent == gameState.getNumAgents():   # wrap back to Pacman
                nextAgent = 0
                newDepth = depth + 1
            else:
                newDepth = depth
            value = max(value, self.miniMax(succState, nextAgent, newDepth))
        return value
    
    def minPlaying(self, gameState: GameState, agentIndex: int, depth: int):
        """
        Helper method for miniMax() that returns the value of a state that minimizes. It will go through all of its successors
        and choose the minimum value as its own value.
        """
        value = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            succState = gameState.generateSuccessor(agentIndex, action)
            nextAgent = agentIndex + 1
            if nextAgent == gameState.getNumAgents():   # wrap back to Pacman
                nextAgent = 0
                newDepth = depth + 1
            else:
                newDepth = depth
            value = min(value, self.miniMax(succState, nextAgent, newDepth))
        return value 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        This method will act as the root node as well as be responsible for initializing alpha-beta values.
        """
        "*** YOUR CODE HERE ***"
        alpha = float('-inf')
        beta = float('inf')
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            succState = gameState.generateSuccessor(0, action)
            value = self.miniMax(succState, 1, 0, alpha, beta)
            if value >= bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)
        return bestAction
        
    def miniMax(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
        """
        A specialized minimax algorithm for the project. It is similar to method with the same name in MinimaxAgent class, but now
        is also taking in alpha-beta values during recursion.
        """
        if (gameState.isWin() or gameState.isLose()) or depth == self.depth:
            return scoreEvaluationFunction(gameState)
        if agentIndex == 0: # Agent is Pacman
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)
    
    def maxValue(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
        """
        Helper method for miniMax() that returns the value of a state that maximizes, while also updating alpha-beta. 
        It will go through all of its successors and choose the maximum value as its own value,
        and also updating alpha to said value.

        While looping through its successors, if the value found is higher than beta (satisfying the condition alpha > beta),
        the method will return the value immediately (pruning).
        """
        value = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            succState = gameState.generateSuccessor(agentIndex, action)
            nextAgent = agentIndex + 1;
            if nextAgent == gameState.getNumAgents():   # wrap back to Pacman
                nextAgent = 0
                newDepth = depth + 1
            else:
                newDepth = depth
            value = max(value, self.miniMax(succState, nextAgent, newDepth, alpha, beta))
            if value > beta:
                return value
            alpha = max(alpha, value)
        return value

    def minValue(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
        """
        Helper method for miniMax() that returns the value of a state that minimizes, while also updating alpha-beta. 
        It will go through all of its successors and choose the minimum value as its own value,
        and also updating beta to said value.

        While looping through its successors, if the value found is smaller than alpha (satisfying the condition alpha > beta),
        the method will return the value immediately (pruning).
        """
        value = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            succState = gameState.generateSuccessor(agentIndex, action)
            nextAgent = agentIndex + 1;
            if nextAgent == gameState.getNumAgents():   # wrap back to Pacman
                nextAgent = 0
                newDepth = depth + 1
            else:
                newDepth = depth
            value = min(value, self.miniMax(succState, nextAgent, newDepth, alpha, beta))
            if value < alpha:
                return value
            beta = min(beta, value)
        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            succState = gameState.generateSuccessor(0, action)
            value = self.expectimax(succState, 1, 0)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
    def expectimax(self, gameState: GameState, agentIndex: int, depth: int):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agentIndex == 0:  # Pacman → maximize
            return self.maxValue(gameState, agentIndex, depth)
        else:  # Ghost → expectation 
            return self.expValue(gameState, agentIndex, depth)

    def maxValue(self, gameState: GameState, agentIndex: int, depth: int):
        """Pacman's best move"""
        value = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            succState = gameState.generateSuccessor(agentIndex, action)
            nextAgent, newDepth = self._nextAgentDepth(agentIndex, depth, gameState)
            value = max(value, self.expectimax(succState, nextAgent, newDepth))
        return value

    def expValue(self, gameState: GameState, agentIndex: int, depth: int):
        legalActions = gameState.getLegalActions(agentIndex)
        prob = 1.0 / len(legalActions)
        value = 0.0
        for action in legalActions:
            succState = gameState.generateSuccessor(agentIndex, action)
            nextAgent, newDepth = self._nextAgentDepth(agentIndex, depth, gameState)
            value += prob * self.expectimax(succState, nextAgent, newDepth)
        return value

    def _nextAgentDepth(self, agentIndex: int, depth: int, gameState: GameState):
        nextAgent = agentIndex + 1
        if nextAgent == gameState.getNumAgents(): 
            return 0, depth + 1
        return nextAgent, depth

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacPos       = currentGameState.getPacmanPosition()
    foodList     = currentGameState.getFood().asList()
    ghostStates  = currentGameState.getGhostStates()
    capsules     = currentGameState.getCapsules()
    walls        = currentGameState.getWalls()
    score        = currentGameState.getScore()

    def bfsMinFoodDist(start, foods):
        """BFS searches for the nearest food along the actual path, not blocked by walls."""
        if not foods:
            return 0
        foodSet = set(foods)
        from util import Queue
        q = Queue()
        q.push((start, 0))
        visited = {start}
        while not q.isEmpty():
            pos, dist = q.pop()
            if pos in foodSet:
                return dist
            x, y = pos
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x+dx, y+dy
                npos = (nx, ny)
                if npos not in visited and not walls[nx][ny]:
                    visited.add(npos)
                    q.push((npos, dist+1))
        return float('inf')

    minFoodDist = bfsMinFoodDist(pacPos, foodList)
    foodScore   = 10.0 / (minFoodDist + 1)  # +1 when standing on food

    ghostScore = 0
    for ghost in ghostStates:
        dist = manhattanDistance(pacPos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            if dist < ghost.scaredTimer:
                ghostScore += 200.0 / (dist + 1)   
            else:
                ghostScore += 5.0 / (dist + 1)     
        else:
            # Active ghost
            if dist == 0:
                ghostScore -= 9999  
            elif dist <= 1:
                ghostScore -= 500
            elif dist <= 3:
                ghostScore -= 150.0 / dist
            else:
                ghostScore -= 3.0 / dist

    foodPenalty    = -5  * len(foodList)
    capsulePenalty = -25 * len(capsules)

    return score + foodScore + ghostScore + foodPenalty + capsulePenalty

# Abbreviation
better = betterEvaluationFunction
