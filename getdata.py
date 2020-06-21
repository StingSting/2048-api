from game2048.game import Game
from game2048.displays import Display,IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent
import numpy as np
import pandas as pd

display1=Display()
display2=IPythonDisplay()



def log2(board):
    for i in range(16):
        if board[i]!=0:
            board[i]=np.log2(board[i])
    return board

for i in range(0,100):
    print(i,"is running")
    game=Game(4,2048,random=False)
    agent=ExpectiMaxAgent(game,display=display2)
    n_iter=0
    max_iter=np.inf
    data=np.zeros((0,17),dtype=float)
    while(n_iter<max_iter) and (not game.end):
        arr1=log2(np.reshape(agent.game.board,newshape=(16,)))
        direction=agent.step()
        arr3=np.hstack((arr1,direction))
        data=np.vstack([data,arr3])
        agent.game.move(direction)
        n_iter+=1
    df=pd.DataFrame(data,columns=None,index=None)
    df.to_csv('/home/huanning/big_project/2048-api-master/data_test.csv',index=0,mode='a',header=0)
        
        
        