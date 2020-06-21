import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

CHANNEL=12

def grid_ohe(input):
    out=[]
    each_c=[]
    for i in range(CHANNEL):
        ret=np.zeros(shape=(4,4),dtype=int)
        for r in range(4):
            for c in range(4):
                if i==input[4*r+c]:
                    ret[r,c]=1
        each_c.append(ret)
    out.append(each_c)
    each_c = []
    return out#shape:4*4*channel

def log2(board):
    for i in range(16):
        if board[i]!=0:
            board[i]=np.log2(board[i])
    return board

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(CHANNEL, 64, kernel_size=(1, 4), padding=(0, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(4, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(4, 4), padding=(2, 2)),
            nn.ReLU()
             )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(128 * 5 * 5),
            nn.Linear(128*5*5, 2048),
            nn.ReLU(),
#            nn.Dropout(0.3),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048,2048),
            # nn.ReLU(),
#            nn.Dropout(0.3),
 #           nn.Dropout(0.5),
 
            nn.BatchNorm1d(2048),
            nn.Linear(2048,256),
            nn.ReLU(),
    #         nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.Linear(256,4)  
    
    #         nn.BatchNorm1d(2048),
    #         nn.Linear(2048,1024),
    #         nn.ReLU(),
    # #         nn.Dropout(0.5),
    #         nn.BatchNorm1d(1024),
    #         nn.Linear(1024,4)           
        )
    def forward(self, x):
        out = self.conv1(x)
        out = out.view(out.shape[0], -1)  # reshape
        out = self.fc(out)
        return out
model = SimpleNet()
#################################
model.load_state_dict(torch.load('model.pkl'))
#################################

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction
    


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class MyOwnAgent(Agent):
    
    def step(self):
        model.eval()
        arr1=log2(np.reshape(self.game.board,newshape=(16,)))
        arr1=grid_ohe(np.int_(arr1))
        arr1 = torch.FloatTensor(arr1)
        # place=[0]#占个位置，凑格式
        # place= torch.LongTensor(place)
        # predict_dataset = data.TensorDataset(arr1,place)
        # predict_loader =data.DataLoader(dataset=predict_dataset,batch_size=1,shuffle=False)
        # for i, (image,label) in enumerate(predict_loader):   
        #         outputs=model(image)
        outputs=model(arr1)
        _,direction=torch.max(outputs.data,1)
        return direction


        # model.eval()
        # arr1=log2(np.reshape(self.game.board,newshape=(16,)))
        # if np.all(self.game.board !=0):
        #     arr1=grid_ohe(np.int_(arr1))
        #     arr1 = torch.FloatTensor(arr1)
        # # place=[0]#占个位置，凑格式
        # # place= torch.LongTensor(place)
        # # predict_dataset = data.TensorDataset(arr1,place)
        # # predict_loader =data.DataLoader(dataset=predict_dataset,batch_size=1,shuffle=False)
        # # for i, (image,label) in enumerate(predict_loader):   
        # #         outputs=model(image)
        #     outputs=model(arr1)
        #     _,direction=torch.max(outputs.data,1)
        #     return direction
        # else:
        #     for i in range(16):
        #         if (i<12)&(arr1[i]==arr1[(i+4)%16]):
        #             return 1
        #         if (i>3)&(arr1[i]==arr1[(i-4)%16]):
        #             return 3
        #         if( (i%4)!=3 )&(arr1[i]==arr1[i+1]):
        #             return 2
        #         else:
        #             return 0
