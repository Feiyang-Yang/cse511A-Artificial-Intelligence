# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    some Directions.X for some X in the set {North, South, West, East, Stop}
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
    score = 0
    
    # get the left beans in game, then the distances
    leftBeans = newFood.asList()
    beansDist = []
    for eachBean in leftBeans:
    	eachBeanDist = manhattanDistance(eachBean,newPos)
    	beansDist.append(eachBeanDist)
    
    # get the distance from ghosts
    ghostDist = []
    for eachGhost in newGhostStates:
    	eachGhostDist = manhattanDistance(eachGhost.getPosition(),newPos)
    	ghostDist.append(eachGhostDist)
    # the closest ghost
    closestGhost = min(ghostDist)
    
    # get the powerful beans
    leftPowBean = []
    leftPowBean = currentGameState.getCapsules()
    
    
    # if already win, add big bonus to distinguish the special status
    if successorGameState.isWin():
    	return float("inf")
    # keep it moving, so big punish on stopping each step
    elif action == Directions.STOP:
    	score -= 5000
    # if not win or stop, the score is based on its current position
    else:
    	# more close to the nearest bean,more bonus,avoid divided by 0
    	# more close to the nearest ghost,more punish,avoid divided by 0
    	score += closestGhost/(min(beansDist)+1)
    	# if it is eating a bean,add bonus as main purpose
    	if newPos in currentGameState.getFood().asList():
    		score *= 2
    	# if it is eating a powerful bean,add bonus
    	if newPos in leftPowBean:
    		score *= 3
    	# if have taken a powerful bean, and it is chasing the ghost
    	if min(newScaredTimes) != 0:
    		score += 500/(closestGhost+1)
    	score += successorGameState.getScore()
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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    def score(gameState, agent_index, depth):
        #reset the agent index to pacman if exceed the total # of agents
        #all the agents moved, so increase the depth
        if agent_index >= num_agents:
            agent_index = pacman_index
            depth += 1
        #base case
        if depth == self.depth:
            return self.evaluationFunction(gameState)
        #continue play
        return play(gameState, agent_index, depth)

    def play(gameState, agent_index, depth):
        #base case
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        #all the legal actions this particular agent can take
        agent_legalActions = gameState.getLegalActions(agent_index)
        scores = []
        #iterate through those actions to find the state that will result lowest score
        for action in agent_legalActions:
            #remove the Directions.STOP action from Pacman's list of possible actions
            if action == "Stop":
                continue
            #next game state
            next_gameState = gameState.generateSuccessor(agent_index, action)
            #next agent index
            next_agent = agent_index + 1
            #get the score of the next agent play
            scores.append(score(next_gameState, next_agent, depth))
        #print "min score is " + str(min(scores))
        if agent_index == pacman_index:
            #pacman's turn to maximize the score
            return max(scores)
        else:
            #ghost's turn to minimize the score
            return min(scores)

    pacman_index = 0
    num_agents = gameState.getNumAgents()
    depth = 0
    score_action_pairs = []
    pacman_legalActions = gameState.getLegalActions(pacman_index)

    #start the game
    for action in pacman_legalActions:
        next_gameState = gameState.generateSuccessor(pacman_index, action)
        next_agent = pacman_index + 1
        s = score(next_gameState, next_agent, depth)
        score_action_pairs.append((s, action))
    #pacman's turn, so choose the highest score and its corresponding action as the optimal action
    (best_score, optimal_action) = max(score_action_pairs)
    print optimal_action
    return optimal_action
	
    	
    

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    def value(gameState, agent_index, depth):
        #reset the agent index to pacman if exceed the total # of agents
        #all the agents moved, so increase the depth
        if agent_index >= num_agents:
            agent_index = pacman_index
            depth += 1
        #base case
        if depth == self.depth:
            return self.evaluationFunction(gameState)
        #continue play
        if agent_index == pacman_index:
            return max_play(gameState, agent_index, depth)
        else:
            return min_play(gameState, agent_index, depth)

    def max_play(gameState, agent_index, depth):
        v = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        agent_legalActions = gameState.getLegalActions(agent_index)
        for action in agent_legalActions:
            #remove the Directions.STOP action from Pacman's list of possible actions
            if action == "Stop":
                continue
            #next game state
            next_gameState = gameState.generateSuccessor(agent_index, action)
            #next agent index
            next_agent = agent_index + 1
            val = value(next_gameState, next_agent, depth)
            if val > v:
                v = val
                alpha = v
            if v > beta:
                break
        return v

    def min_play(gameState, agent_index, depth):
        v = float('inf')
        alpha = -float('inf')
        beta = float('inf')
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        agent_legalActions = gameState.getLegalActions(agent_index)
        for action in agent_legalActions:
            #remove the Directions.STOP action from Pacman's list of possible actions
            if action == "Stop":
                continue
            #next game state
            next_gameState = gameState.generateSuccessor(agent_index, action)
            #next agent index
            next_agent = agent_index + 1
            val = value(next_gameState, next_agent, depth)
            if val < v:
                v = val
                beta = v
            if v < alpha:
                break
        return v
    pacman_index = 0
    num_agents = gameState.getNumAgents()
    depth = 0
    score_action_pairs = []
    pacman_legalActions = gameState.getLegalActions(pacman_index)
    v = -float('inf')
    alpha = -float('inf')
    beta = float('inf')
    optimal_action = "Stop"
    #start the game
    for action in pacman_legalActions:
        next_gameState = gameState.generateSuccessor(pacman_index, action)
        next_agent = pacman_index + 1
        val = value(next_gameState, next_agent, depth)
        if val > v:
            v = val
            alpha = v
            optimal_action = action
        if v > beta:
            break
    return optimal_action

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
    def expect(state,curDepth,curAgent):
    	expectV = 0
    	add = 0
    	ghostTotal = state.getNumAgents() - 1
    	# base case
    	if state.isWin() or state.isLose() or curDepth == self.depth:
    		return self.evaluationFunction(state)
    	# recursive rule for average expect
    	if curAgent == ghostTotal:
    		for eachAct in state.getLegalActions(curAgent):
    			add = maxVal(state.generateSuccessor(curAgent,eachAct), curDepth)
    			expectV += add
    	else:
    		for eachAct in state.getLegalActions(curAgent):
    			add = expect(state.generateSuccessor(curAgent,eachAct), curDepth,1+curAgent)
    			expectV += add
    	expectV = expectV/ghostTotal
    	return expectV
    
    def maxVal(state,curDepth):
    	# base case
    	if state.isWin() or state.isLose() or curDepth == self.depth:
    		return self.evaluationFunction(state)
    	values = []
    	for eachAct in state.getLegalActions(0):
    		compareV = expect(state.generateSuccessor(0,eachAct), curDepth,1)
    	return max(values)
    
    vals = []
    legalActs = gameState.getLegalActions(0)
    for eachAct in legalActs:
    	expectVal = expect(gameState.generateSuccessor(0, eachAct), self.depth, 1)
    	vals.append(expectVal)
    maxValue = max(vals)
    for eachAct in legalActs:
    	expectVal = expect(gameState.generateSuccessor(0, eachAct), self.depth, 1)
    	if maxValue == expectVal:
    		return eachAct
    	
    	
    
    
def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  foods = currentGameState.getFood().asList()
  ghost_states = currentGameState.getGhostStates()
  capsules = currentGameState.getCapsules()
  pacman_position = currentGameState.getPacmanPosition()
  #i = 0
  capsule_score = 0
  #num_capsules = len(capsules)
  for capsule in capsules:
    capsule_dist = manhattanDistance(pacman_position, capsule)
    capsule_score += 20/(capsule_dist+1)

  ghost_score = 0
  #num_scared_ghost = 0
  for ghost_state in ghost_states:
    ghost_dist = manhattanDistance(pacman_position, ghost_state.getPosition())
    if ghost_state.scaredTimer > 0:
        #num_scared_ghost += 1
        ghost_score += 200 / (ghost_dist+1)
    else:
        if ghost_dist < 2:
            ghost_score += -500 / (ghost_dist+1)
  #   print "ghost "+str(i)+" has "+str(gs.scaredTimer)+" left"
  #   i+=1
  # print "next"
  pacman_position = currentGameState.getPacmanPosition()
  food_dist = []
  #num_foods = len(foods)
  for food in foods:
    food_dist.append(manhattanDistance(pacman_position, food))
  if not food_dist:
    food_dist.append(0)
  food_score = 10 / (min(food_dist)+1)
  final_score = currentGameState.getScore() + capsule_score + ghost_score + food_score
  return final_score


# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

