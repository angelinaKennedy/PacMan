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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        food = [manhattanDistance(newPos, foodNear) for foodNear in newFood.asList()]
        if food:
            foodDistance = min(food)
            score += 1.0 / foodDistance

        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            if scaredTime == 0 and manhattanDistance(newPos, ghostPos) < 2:
                score -= 1000

        return score

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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

        def maxValue(state, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            value = float('-inf')
            legalActions = state.getLegalActions(0)

            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                value = max(value, minValue(successor, depth, 1))

            return value

        def minValue(state, depth, ghostIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            value = float('inf')
            legalActions = state.getLegalActions(ghostIndex)

            for action in legalActions:
                successor = state.generateSuccessor(ghostIndex, action)

                if ghostIndex == state.getNumAgents() - 1:
                    value = min(value, maxValue(successor, depth - 1))
                else:
                    value = min(value, minValue(successor, depth, ghostIndex + 1))

            return value

        legalActions = gameState.getLegalActions(0)
        bestAction = Directions.STOP
        bestValue = float('-inf')

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = minValue(successor, self.depth, 1)

            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def maxValue(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            value = float('-inf')
            legalActions = state.getLegalActions(0)

            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                value = max(value, minValue(successor, depth, 1, alpha, beta))

                if value > beta:
                    return value

                alpha = max(alpha, value)

            return value

        def minValue(state, depth, ghostIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            value = float('inf')
            legalActions = state.getLegalActions(ghostIndex)

            for action in legalActions:
                successor = state.generateSuccessor(ghostIndex, action)

                if ghostIndex == state.getNumAgents() - 1:
                    value = min(value, maxValue(successor, depth - 1, alpha, beta))
                else:
                    value = min(value, minValue(successor, depth, ghostIndex + 1, alpha, beta))

                if value < alpha:
                    return value

                beta = min(beta, value)

            return value

        legalActions = gameState.getLegalActions(0)
        bestAction = Directions.STOP
        alpha = float('-inf')
        beta = float('inf')

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = minValue(successor, self.depth, 1, alpha, beta)

            if value > alpha:
                alpha = value
                bestAction = action

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxValue(state, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            value = float('-inf')
            legalActions = state.getLegalActions(0)

            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                value = max(value, expectiValue(successor, depth, 1))

            return value

        def expectiValue(state, depth, ghostIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            value = 0.0
            legalActions = state.getLegalActions(ghostIndex)
            num = len(legalActions)

            for action in legalActions:
                successor = state.generateSuccessor(ghostIndex, action)

                if ghostIndex == state.getNumAgents() - 1:
                    value += maxValue(successor, depth - 1)
                else:
                    value += expectiValue(successor, depth, ghostIndex + 1)

            return value / num

        legalActions = gameState.getLegalActions(0)
        bestAction = Directions.STOP
        bestValue = float('-inf')

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = expectiValue(successor, self.depth, 1)

            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This function considers:
    - The current score: To encourages Pacman to seek higher scores.
    - The distance to the nearest food pellet
    - The distance to the nearest ghost: To avoid ghosts, with a higher penalty if they are close.
    - The number of remaining capsules: To prioritize eating capsules for potential ghost-scaring.

    The weights for each factor are determined empirically.
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    if food:
        foodDistance = min([manhattanDistance(pacmanPosition, pellet) for pellet in food])
    else:
        foodDistance = 0

    ghostDistance = min([manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates])
    if ghostDistance > 0:
        ghostDistance = 1.0 / ghostDistance

    remainingCap = len(capsules)

    scoreWeight = 100.0
    foodWeight = -10.0
    ghostWeight = -50.0
    capsuleWeight = -20.0

    evaluation = scoreWeight * score + foodWeight * foodDistance + ghostWeight * ghostDistance + capsuleWeight * remainingCap

    return evaluation

# Abbreviation
better = betterEvaluationFunction
