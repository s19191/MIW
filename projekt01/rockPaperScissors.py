import numpy as np
import matplotlib.pyplot as plt

print("Legend:")
print("R - Rock")
print("P - Paper")
print("S - Scissors")

start = ["R", "P", "S"]
responses = {"R": "P", "P": "S", "S": "R"}
probability = np.array([[1 / 3, 1 / 3, 1 / 3],
                        [1 / 3, 1 / 3, 1 / 3],
                        [1 / 3, 1 / 3, 1 / 3]])
counter = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]])

#RR RP RS
#PR PP PS
#SR SP SS

computerScoreList = []
computerScore = 0
totalGamesPlayed = 0

while True:
    print("******** Game number: ", totalGamesPlayed + 1, " ********")
    previousMove = "R"
    predictMove = ""
    if previousMove == "R":
        predictMove = np.random.choice(start, p=probability[0])
    elif previousMove == "P":
        predictMove = np.random.choice(start, p=probability[1])
    elif previousMove == "S":
        predictMove = np.random.choice(start, p=probability[2])

    computersMove = responses.get(predictMove)

    opponentsMove = input("Your move (R/P/S): ").upper()

    if not start.__contains__(opponentsMove):
        print("Invalid value, try again! You should type R - rock/ P - paper/ S - scissors!")
        continue
    else:
        print("Computers move: ", computersMove)
        if predictMove == opponentsMove:
            print("Computer won!")
            computerScore += 1
        elif computersMove == opponentsMove:
            print("Draw!")
        else:
            print("You won!")
            computerScore -= 1

        counter[start.index(previousMove)][start.index(opponentsMove)] += 1
        row = start.index(previousMove)

        for i in range(3):
            probability[row][i] = counter[row][i] / np.sum(counter[row])

        # for i in counter:
        #     print(i)
        # for i in probability:
        #     print(i)

        previousMove = opponentsMove
        totalGamesPlayed += 1
        computerScoreList.append(computerScore)

    ifEnd = input("Do you want to end the game? (YES - y, NO - any other key)")
    if ifEnd.lower() == "y":
        break

print("******** End of the game! ********")
print("Total games played: ", totalGamesPlayed)
plt.clf()
plt.plot(np.arange(totalGamesPlayed), computerScoreList, "r*")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("Computer score")
plt.show()