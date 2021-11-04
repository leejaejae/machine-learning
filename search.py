# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.


import inspect
import sys
import random
from collections import deque
from queue import PriorityQueue


def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        pass

    def isGoalState(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        pass

    def getSuccessors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        pass

    def getCostOfActions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        pass


def random_search(problem):
    """
    Search the nodes in the search tree randomly.

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm.

    This random_search function is just example not a solution.
    You can write your code by examining this function
    """
    start = problem.getStartState()
    node = [(start, "", 0)]   # class is better

    frontier = [node]

    explored = set()
    while frontier:
        node = random.choice(frontier)
        state = node[-1][0]

        if problem.isGoalState(state):  # path
            return [x[1] for x in node][1:]

        if state not in explored:
            explored.add(state)

            for successor in problem.getSuccessors(state):
                if successor[0] not in explored:
                    parent = node[:]
                    parent.append(successor)
                    frontier.append(parent)
    return []


def depth_first_search(problem):
    """Search the deepest nodes in the search tree first."""
    start = problem.getStartState()
    node = [(start, "", 0)]

    frontier = [node]

    visited = set()

    while frontier:  # while stack is not empty do
        node = frontier.pop()
        state = node[-1][0]

        if problem.isGoalState(state):  # path
            return [x[1] for x in node][1:]

        if state not in visited:  # if node is not labeled as visit then
            visited.add(state)  # label node as visit

            for successor in problem.getSuccessors(state):
                if successor[0] not in visited:
                    parent = node[:]
                    parent.append(successor)
                    frontier.append(parent)


def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    start = problem.getStartState()
    node = [(start, "", 0)]

    frontier = deque([node])

    visited = set()

    while frontier:  # while queue is not empty do
        node = frontier.popleft()  # popleft: pop의 반대, 왼쪽(앞쪽)에서부터 차례대로 제거 반환
        state = node[-1][0]

        if problem.isGoalState(state):  # path
            return [x[1] for x in node][1:]

        if state not in visited:  # if node is not labeled as visit then
            visited.add(state)  # label node as visit

            for successor in problem.getSuccessors(state):
                if successor[0] not in visited:
                    parent = node[:]
                    parent.append(successor)
                    frontier.append(parent)


class Node:
    def __init__(self, cost=0, path="", state=None):
        self.cost = cost
        self.path = path
        self.state = state

    def __lt__(self, other):  # <
        return self.cost < other.cost

    def __le__(self, other):  # <=
        return self.cost <= other.cost

    def __gt__(self, other):  # >
        return self.cost > other.cost

    def __ge__(self, other):  # >=
        return self.cost >= other.cost


def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    start = problem.getStartState()
    node = [Node(0, "", start)]

    frontier = PriorityQueue()
    frontier.put(node)

    visited = set()

    while frontier:
        node = frontier.get()
        state = node[0].state
        cost = node[0].cost

        if problem.isGoalState(state):  # path
            return [x.path for x in node][:-1][::-1]

        if state not in visited:  # if node is not labeled as visit then
            visited.add(state)
            for successor in problem.getSuccessors(state):
                if successor[0] not in visited:
                    total_cost = cost + successor[2]
                    parent = node[:]
                    parent.insert(0, Node(total_cost, successor[1], successor[0]))
                    frontier.put(parent)


def heuristic_manhattan(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem. This heuristic is trivial.
    """
    cost = 0
    s = state.cells
    goal = [(i, j) for i in range(3) for j in range(3)]
    cost = sum(abs(goal[s[i][j]][0]-i) + abs(goal[s[i][j]][1]-j) for i in range(3) for j in range(3))

    return cost


def aStar_search(problem, heuristic=heuristic_manhattan):
    """Search the node that has the lowest combined cost and heuristic first."""
    start = problem.getStartState()
    node = [Node(0, "", start)]

    frontier = PriorityQueue()
    frontier.put(node)

    visited = set()

    while frontier:
        node = frontier.get()
        state = node[0].state
        cost = node[0].cost

        manhattan = heuristic(state)

        if problem.isGoalState(state):  # path
            return [x.path for x in node][:-1][::-1]

        if state not in visited:  # if node is not labeled as visit then
            visited.add(state)

            for successor in problem.getSuccessors(state):
                if successor[0] not in visited:
                    total_cost = cost + successor[2]
                    parent = node[:]
                    parent.insert(0, Node(manhattan + total_cost, successor[1], successor[0]))
                    frontier.put(parent)



# Abbreviations
rand = random_search
bfs = breadth_first_search
dfs = depth_first_search
astar = aStar_search
ucs = uniform_cost_search
