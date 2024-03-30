# factorOperations.py
# -------------------
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

from typing import List
from bayesNet import Factor
import functools
from util import raiseNotDefined


def joinFactorsByVariableWithCallTracking(callTrackingList=None):
    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that
        contain that variable.

        Returns a tuple of
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin = [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len(
            [factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +
                             "joinVariable: " + str(joinVariable) + "\n" +
                             ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))

        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable


joinFactorsByVariable = joinFactorsByVariableWithCallTracking()


########### ########### ###########
########### QUESTION 2  ###########
########### ########### ###########

def joinFactors(factors: List[Factor]):
    """
    Input factors is a list of factors.

    You should calculate the set of unconditioned variables and conditioned
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input
    (such as getProbability and setProbability) can handle
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                             + "unconditionedVariables: " + str(intersect) +
                             "\nappear in more than one input factor.\n" +
                             "Input factors: \n" +
                             "\n".join(map(str, factors)))

    "*** YOUR CODE HERE ***"
    #Your joinFactors should return a new Factor.
    #For a general joinFactors operation, which variables are unconditioned in the returned Factor? Which variables are conditioned?
    # Factor.variableDomainsDict=> maps each variable to a list of values that it can take on (its domain).
    # A Factor gets its variableDomainsDict from the BayesNet from which it was instantiated=> (i.e. inputVariableDomains in init of factors
    # As a result, it contains all the variables of the BayesNet, not only the unconditioned and conditioned variables used in the Factor. i need to use  Factor.unconditionedVariables & Factor.conditionedVariables
    # For this problem, you may assume that all the input Factors have come from the same BayesNet, and so their variableDomainsDicts are all the same.
    """ Factor A: P(X, Y)=> P(X=0, Y=0) = 0.1 
                            P(X=1, Y=0) = 0.2
                            P(X=0, Y=1) = 0.3 
                            P(X=1, Y=1) = 0.4
                            
         Factor B: P(Y, Z)=> P(Y=0, Z=0) = 0.5 
                             P(Y=1, Z=0) = 0.6 
                             P(Y=0, Z=1) = 0.7 
                             P(Y=1, Z=1) = 0.8 
        Resultant Factor (P(X, Y, Z)):

        P(X=0, Y=0, Z=0) = P(X=0, Y=0) * P(Y=0, Z=0) = 0.1 * 0.5 = 0.05     X=0 and Z=0 given Y=0 
        P(X=1, Y=0, Z=0) = P(X=1, Y=0) * P(Y=0, Z=0) = 0.2 * 0.5 = 0.1      X=1 and Z=0 given Y=0
        P(X=0, Y=1, Z=0) = P(X=0, Y=1) * P(Y=1, Z=0) = 0.3 * 0.6 = 0.18     X=0 and Z=0 given Y=1
        P(X=1, Y=1, Z=0) = P(X=1, Y=1) * P(Y=1, Z=0) = 0.4 * 0.6 = 0.24     X=1 and Z=0 given Y=1
        P(X=0, Y=0, Z=1) = P(X=0, Y=0) * P(Y=0, Z=1) = 0.1 * 0.7 = 0.07     X=0 and Z=1 given Y=0
        P(X=1, Y=0, Z=1) = P(X=1, Y=0) * P(Y=0, Z=1) = 0.2 * 0.7 = 0.14     X=1 and Z=1 given Y=0
        P(X=0, Y=1, Z=1) = P(X=0, Y=1) * P(Y=1, Z=1) = 0.3 * 0.8 = 0.24     X=0 and Z=1 given Y=1
        P(X=1, Y=1, Z=1) = P(X=1, Y=1) * P(Y=1, Z=1) = 0.4 * 0.8 = 0.32     X=1 and Z=1 given Y=1
        
        ie P(X,Z|Y)
        The join operation, let's say joinFactors(A, B), would result in a factor where:
        Unconditioned variables are {X, Y, Z}.
        Conditioned variables are {Y}.
        
        """
    inputUnconditionedVariables = set()
    CondFactors = set()
    #take the conditioned/unconditioned variable from each factor on the list
    for factor in factors:
        inputUnconditionedVariables.update(factor.unconditionedVariables())
        CondFactors.update(factor.conditionedVariables())

    inputConditionedVariables = CondFactors.difference(inputUnconditionedVariables)
    inputVariableDomainsDict=list(factors)[0].variableDomainsDict()
    #print(list(factors)[0])
    #print(inputUnconditionedVariables)
    #print(inputConditionedVariables)

    FinalFactor = Factor(inputUnconditionedVariables, inputConditionedVariables, inputVariableDomainsDict) #(inputUnconditionedVariables, inputConditionedVariables, inputVariableDomainsDict):

    # Join Tables Through Multiplication
    for Dict in FinalFactor.getAllPossibleAssignmentDicts():
        #print(factor.getProbability(Dict))
        prob = 1
        for factor in factors:
            prob = prob * factor.getProbability(Dict)
        FinalFactor.setProbability(Dict, prob)

    return FinalFactor
    "*** END YOUR CODE HERE ***"


########### ########### ###########
########### QUESTION 3  ###########
########### ########### ###########

def eliminateWithCallTracking(callTrackingList=None):
    def eliminate(factor: Factor, eliminationVariable: str):
        """
        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.

        You should calculate the set of unconditioned variables and conditioned
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                             + "in this factor\n" +
                             "eliminationVariable: " + str(eliminationVariable) + \
                             "\nunconditionedVariables:" + str(factor.unconditionedVariables()))

        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                             + "can't eliminate \nthat variable.\n" + \
                             "eliminationVariable:" + str(eliminationVariable) + "\n" + \
                             "unconditionedVariables: " + str(factor.unconditionedVariables()))

        "*** YOUR CODE HERE ***"
        #Implement the eliminate function in factorOperations.py.
        # It takes a Factor and a variable to eliminate and returns a new Factor that does not contain that variable.
        # This corresponds to summing all of the entries in the Factor which only differ in the value of the variable being eliminated.
        unconditionedvars = factor.unconditionedVariables()
        unconditionedvars.remove(eliminationVariable)
        FinalFactor = Factor(unconditionedvars, factor.conditionedVariables(), factor.variableDomainsDict())
        #print(FinalFactor.getAllPossibleAssignmentDicts())
        for NewProbDict in FinalFactor.getAllPossibleAssignmentDicts():
            #dict={'D': 'wet'}
            TotalProb = 0
            #print(factor.getAllPossibleAssignmentDicts())
            for OldProbDict in factor.getAllPossibleAssignmentDicts():
                IsConnected = True #is there conditional dependencies between variables represented in the network?
                #print("HI")
                #print(NewProbDict.keys())
                #print(OldProbDict.keys())
                #print("BYE")
                for key in NewProbDict.keys():
                    Exist = key in OldProbDict.keys()
                    Equal = OldProbDict[key] == NewProbDict[key]
                    notCon=not (Exist and Equal)
                    if notCon:
                        IsConnected = False

                if IsConnected==True:
                    if len(NewProbDict) <= len(OldProbDict):
                        TotalProb += factor.getProbability(OldProbDict)
            FinalFactor.setProbability(NewProbDict, TotalProb)
        return FinalFactor
        "*** END YOUR CODE HERE ***"

    return eliminate


eliminate = eliminateWithCallTracking()

