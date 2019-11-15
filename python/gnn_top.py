import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
import itertools

class GraphNet(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden, De, Do, 
                 fr_activation=0, fo_activation=0, fc_activation=0, optimizer = 0, verbose = False, sum_O=False):
        super(GraphNet, self).__init__()
        self.hidden = hidden
        self.P = len(params)
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.n_targets = n_targets
        self.fr_activation = fr_activation
        self.fo_activation = fo_activation
        self.fc_activation = fc_activation
        self.optimizer = optimizer
        self.verbose = verbose
        self.assign_matrices()

        self.sum_O = sum_O
        self.Ra = torch.ones(self.Dr, self.Nr)

        self.activations = nn.ModuleList([nn.ReLU(),nn.ELU(),nn.SELU()])

        self.fr = nn.Sequential(nn.Linear(2 * self.P + self.Dr, self.hidden),
                                self.activations[self.fr_activation],
                                nn.Linear(self.hidden, int(self.hidden/2)),
                                self.activations[self.fr_activation],
                                nn.Linear(int(self.hidden/2), self.De),
                                self.activations[self.fr_activation]).cuda()

        #self.fr2 = nn.Sequential(nn.Linear(2 * (self.P + self.De) + self.Dr, self.hidden),
        #                         self.activations[self.fr_activation],
        #                         nn.Linear(self.hidden, int(self.hidden/2)),
        #                         self.activations[self.fr_activation],
        #                         nn.Linear(int(self.hidden/2), self.De),
        #                         self.activations[self.fr_activation]).cuda()
        
        self.fo = nn.Sequential(nn.Linear(self.P + self.Dx + self.De, self.hidden),
                                self.activations[self.fo_activation],
                                nn.Linear(self.hidden, int(self.hidden/2)),
                                self.activations[self.fo_activation],
                                nn.Linear(int(self.hidden/2), self.Do),
                                self.activations[self.fo_activation]).cuda()
        if self.sum_O:
            self.fc = nn.Sequential(nn.Linear(self.Do, self.hidden),
                                    self.activations[self.fc_activation],
                                    nn.Linear(self.hidden, int(self.hidden/2)),
                                    self.activations[self.fc_activation],
                                    nn.Linear(int(self.hidden/2), self.n_targets)).cuda()
        else:
            self.fc = nn.Sequential(nn.Linear(self.Do * self.N, self.hidden),
                                    self.activations[self.fc_activation],
                                    nn.Linear(self.hidden, int(self.hidden/2)),
                                    self.activations[self.fc_activation],
                                    nn.Linear(int(self.hidden/2), self.n_targets)).cuda()

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = self.Rr.cuda()
        self.Rs = self.Rs.cuda()

    def edge_conv(self, x, f, num_edge_feat):
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        B = torch.transpose(B, 1, 2).contiguous()
        E = f(B.view(-1, 2 * num_edge_feat + self.Dr)).view(-1, self.Nr, self.De)
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        return Ebar
        
    def forward(self, x):
        ### First MLP (Message Passing) ###
        Ebar = self.edge_conv(x, self.fr, self.P)

        C = torch.cat([x, Ebar], 1)

        #Ebar2 = self.edge_conv(C, self.fr2, self.P+self.De)
        #del Ebar
        #del C
        #C2 = torch.cat([x, Ebar2], 1)
        #C2 = torch.transpose(C2, 1, 2).contiguous()

        ### Second MLP ###
        O = self.fo(C.view(-1, self.P + self.Dx + self.De)).view(-1, self.N, self.Do)
        #del Ebar2
        #del C2

        ### Sum over the O matrix ###
        ### Classification MLP ###
        if self.sum_O:
            O = torch.sum(O, dim=1)
            N = self.fc(O.view(-1, self.Do))
        else:
            N = self.fc(O.view(-1, self.Do * self.N))
        del O

        return N

    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])

####################

class GraphNetOld(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden, De, Do, 
                 fr_activation=0, fo_activation=0, fc_activation=0, optimizer = 0, verbose = False, sum_O = False):
        super(GraphNetOld, self).__init__()
        self.hidden = hidden
        self.P = len(params)
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.n_targets = n_targets
        self.fr_activation = fr_activation
        self.fo_activation = fo_activation
        self.fc_activation = fc_activation
        self.optimizer = optimizer
        self.verbose = verbose
        self.assign_matrices()

        self.sum_O = sum_O
        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden).cuda()
        self.fr2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3 = nn.Linear(int(self.hidden/2), self.De).cuda()
        self.fo1 = nn.Linear(self.P + self.Dx + self.De, self.hidden).cuda()
        self.fo2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fo3 = nn.Linear(int(self.hidden/2), self.Do).cuda()
        if self.sum_O:
            self.fc1 = nn.Linear(self.Do *1, self.hidden).cuda()
        else:
            self.fc1 = nn.Linear(self.Do * self.N, self.hidden).cuda()
        self.fc2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fc3 = nn.Linear(int(self.hidden/2), self.n_targets).cuda()

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = self.Rr.cuda()
        self.Rs = self.Rs.cuda()

    def forward(self, x):
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        if self.fr_activation ==2:
            B = nn.functional.selu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.selu(self.fr2(B))
            E = nn.functional.selu(self.fr3(B).view(-1, self.Nr, self.De))            
        elif self.fr_activation ==1:
            B = nn.functional.elu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.elu(self.fr2(B))
            E = nn.functional.elu(self.fr3(B).view(-1, self.Nr, self.De))
        else:
            B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.relu(self.fr2(B))
            E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        C = torch.cat([x, Ebar], 1)
        del Ebar
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        if self.fo_activation ==2:
            C = nn.functional.selu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.selu(self.fo2(C))
            O = nn.functional.selu(self.fo3(C).view(-1, self.N, self.Do))
        elif self.fo_activation ==1:
            C = nn.functional.elu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.elu(self.fo2(C))
            O = nn.functional.elu(self.fo3(C).view(-1, self.N, self.Do))
        else:
            C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.relu(self.fo2(C))
            O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C
        ## sum over the O matrix
        if self.sum_O:
            O = torch.sum( O, dim=1)
        ### Classification MLP ###
        if self.fc_activation ==2:
            if self.sum_O:
                N = nn.functional.selu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.selu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.selu(self.fc2(N))       
        elif self.fc_activation ==1:
            if self.sum_O:
                N = nn.functional.elu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.elu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.elu(self.fc2(N))
        else:
            if self.sum_O:
                N = nn.functional.relu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.relu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.relu(self.fc2(N))
        del O
        #N = nn.functional.relu(self.fc3(N))
        N = self.fc3(N)
        return N

    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])

####################
