import numpy as np
import scipy as sp

import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import IPython.display

import copy

import torch
from torch import nn, tensor
from torch.autograd import Function
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import transpile, assemble
#from qiskit.visualization import *

import sys

matplotlib.rcParams['animation.embed_limit'] = 100

SEED = int(sys.argv[1])

torch.manual_seed(SEED)
np.random.seed(SEED)

directions = [[1,0], [0,1], [-1,0], [0,-1]]

###############################################
###############################################
################## PARAMETERS #################
###############################################
###############################################

#PREDATOR PARAMETERS
TWEAK_PROB = .2                  #Probability that a weight is modified
TWEAK_SIZE = .05                 #Amount that the weight gets modified by
FORWARD_BIAS = 0                 #Bias introduced to have the organisms tend to move forward
VISION_RADIUS = 4                #Radius of the feild of view for each predator
FOOD_REGENERATION = 50           #Amount of time a predator can survive for since last time eaten
AMOUNT_OF_FOOD_TO_REPRODUCE = 3  #Amount of food needed to reproduce
INITIAL_REGENERATION = 80        #Amount of time a predator has to live (without eating) initially
NUMBER_OF_CHILDREN = 1           #Number of children a predator spawns when 

#SIMULATION PARAMETERS
ANIMATION_LENGTH = 1000
CLUMP_SIZE = 6
INITIAL_NUMBER_OF_FOOD_CLUMPS = 20
PROBABILITY_OF_NEW_CLUMP = .08
NUM_QUANTUM_PREDATORS = 20
NUM_CLASSICAL_PREDATORS = 20
size = 100

###############################################
###############################################
#---------------------------------------------#
###############################################
###############################################


class Predator:
    """
    Class which embodies the predator. This class allows the predators to spawn,
    move,and reproduce. 
    """
    
    vision_radius = VISION_RADIUS
    
    def __init__(self, weights, pos, direction, quantum = False):
        self.food = 0
        self.pos = pos
        self.dir = [1,0] #(1,0) = down, (-1, 0) = up, (0, -1) = left, (0,1) = right
        self.time_until_death = INITIAL_REGENERATION
        self.quantum = quantum
        
        #Create the neural network and upload weights if reproduced from other predator
        if not self.quantum:
            self.model = net((2*self.vision_radius + 1)**2, 3)
        else:
            self.model = QuantNet((2*self.vision_radius + 1)**2)
        if weights != 0:
            self.model.load_state_dict(weights)
        self.weights = self.model.state_dict()
        
    def reproduce(self, world):
        
        #Copy over weights to children with slight modifications (mutations)
        new_weights = copy.deepcopy(self.weights)
        for key in new_weights.keys():
            params = new_weights[key]
            shape = params.size()
            params_flattened = torch.flatten(params)
            for i in range(len(params_flattened)):
                if np.random.rand() < TWEAK_PROB:
                    params_flattened[i] = params_flattened[i] + np.random.normal(0, TWEAK_SIZE)
            params = torch.reshape(params_flattened, shape)
        
        #Children end up facing a random direction and spawn near their parent
        reproduced_position = [np.random.randint(-2, 3) + self.pos[0], np.random.randint(-2, 3)+ self.pos[1]]
        while (reproduced_position[0]<0 or reproduced_position[0]>=len(world)) or (reproduced_position[1]<0 or reproduced_position[1]>=len(world)):
            reproduced_position = [np.random.randint(-2, 3) + self.pos[0], np.random.randint(-2, 3)+ self.pos[1]]
        direction = [[1,0], [-1, 0], [0,1], [0,-1]][np.random.randint(0,4)]
        
        return Predator(new_weights, reproduced_position, direction, quantum = self.quantum)
        
    def move(self, world):
        
        #Get the field of view
        field_of_view = np.zeros((self.vision_radius*2 + 1, self.vision_radius*2 + 1))
        for i in range(len(field_of_view)):
            for j in range(len(field_of_view[i])):
                if (i + self.pos[0] - self.vision_radius < 0) or (i + self.pos[0] - self.vision_radius >= len(world)):
                    field_of_view[i][j] = -1
                elif (j + self.pos[1] - self.vision_radius < 0) or (j + self.pos[1] - self.vision_radius >= len(world)):
                    field_of_view[i][j] = -1
                else:
                    field_of_view[i][j] = world[i + self.pos[0] - self.vision_radius, j + self.pos[1] - self.vision_radius]
        if self.dir == [1,0]: #pointing down means need to rotate by 180
            field_of_view = np.rot90(field_of_view, k=2)
        if self.dir == [0, -1]: #pointing left means need to rotate right
            field_of_view = np.rot90(field_of_view, k=-1)
        if self.dir == [0, 1]: #pointing right means need to rotate left
            field_of_view = np.rot90(field_of_view, k=1)
        
        #Get p_moves from the neural network
        #p_moves[0] = weight on moving forward
        #p_moves[1] = weight on moving forward and turning left
        #p_moves[2] = weight on moving forward and turning right
        vision_tensor = tensor([field_of_view.flatten()], dtype=torch.float32)
        if not self.quantum:
            p_moves = self.model(vision_tensor).detach().numpy()[0] #WORKS FOR CLASSICAL
        else:
            p_moves = np.reshape(self.model(vision_tensor).detach().numpy(), (1,3))[0]
        p_moves[0] += FORWARD_BIAS
        
        #Move forward if able to
        can_move = True
        if (self.pos[0]+self.dir[0]<0) or (self.pos[0]+self.dir[0]>=len(world)) or (self.pos[1]+self.dir[1]<0) or (self.pos[1]+self.dir[1]>=len(world)):
            can_move = False
        if(can_move):
            self.pos[0] = self.pos[0]+self.dir[0]
            self.pos[1] = self.pos[1]+self.dir[1]
        
        #Get the highest weighted move and excecute the move
        best_move = np.argmax(p_moves)
        if best_move == 1:
            if self.dir==[1,0]: #down --> right
                self.dir=[0,1]
            elif self.dir==[0,1]: #right --> up
                self.dir=[-1,0] 
            elif self.dir==[-1,0]: #up --> left
                self.dir=[0,-1]
            elif self.dir==[0, -1]:
                self.dir=[1,0]  #left --> down
        elif best_move == 2:
            if self.dir==[1,0]: #down --> left
                self.dir=[0,-1]
            elif self.dir==[0,1]: #right --> down
                self.dir=[1,0] 
            elif self.dir==[-1,0]: #up --> right
                self.dir=[0,1]
            elif self.dir==[0, -1]:
                self.dir=[-1,0]  #left --> up
        
        #AUX print statements
        #dir_dic = {(1,0):"down", (-1, 0):"up", (0, -1):"left", (0,1):"right"}
        #print("At position:  ", self.pos)
        #print("Facing direction: ", dir_dic[tuple(self.dir)])
        #print(field_of_view)
        #print("Food is: ", self.food)
        #print()
        
class net(nn.Module):
  """
  Class which generates the classical neural network which
  the predators use. The format is a standard feed forward
  network with two hidden layers which each half the size 
  of the previous layers
  """
  def __init__(self,input_size,output_size):
    super(net,self).__init__()
    self.l1 = nn.Linear(input_size,int(input_size/2))
    self.l2 = nn.Linear(int(input_size/2),int(input_size/4))
    self.l3 = nn.Linear(int(input_size/4),output_size)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    
    self.apply(self._init_weights)
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=1.0)
        module.bias.data.zero_() 
  def forward(self,x):
    output = self.l1(x) 
    output = self.relu(output)
    output = self.l2(output)
    output = self.l3(output)
    output = torch.abs(output)
    output = output/100   #Scale output so each weight is roughly in the range (0,1)
    return output

class QuantumCircuit:
    """
    Class which creates the quantum circuit needed to 
    create the hybrid neural network. Methods include
    initialization and running the circuit (taking inputs
    and then returning the corresponding outputs)
    """
    def __init__(self, n_qubits, backend, shots):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        
        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')
        
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
    
    def run(self, thetas):
        t_qc = transpile(self._circuit,
                         self.backend)
        qobj = assemble(t_qc,
                        shots=self.shots,
                        parameter_binds = [{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        
        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)
        
        return np.array([expectation])
    
class HybridFunction(Function):
    """
    Function which uses the quantum circuit above to take in
    a pytorch tensor, feed it through the quantum circuit and
    then extract the output as a pytorch tensor
    """
    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z1 = ctx.quantum_circuit.run([input[0].tolist()[0]])
        expectation_z2 = ctx.quantum_circuit.run([input[0].tolist()[1]])
        expectation_z3 = ctx.quantum_circuit.run([input[0].tolist()[2]])
        result = torch.tensor([expectation_z1, expectation_z2, expectation_z3])
        #ctx.save_for_backward(input, result)

        return result
    
class Hybrid(nn.Module):
    """
    Creates a hybird layer from the hybrid function
    and declares the feed forward nature of the layer
    """
    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(3, backend, shots)
        self.shift = shift
        
    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)
    

class QuantNet(nn.Module):
    """
    Class which generates the classical neural network which
    the predators use. The format is a standard feed forward
    network with two hidden layers which each half the size 
    of the previous layers. Additionally, the quantum circuit 
    is added to the end using the hybrid layer
    """
    def __init__(self,input_size):
        super(QuantNet,self).__init__()
        self.l1 = nn.Linear(input_size,int(input_size/2))
        self.l2 = nn.Linear(int(input_size/2),int(input_size/4))
        self.l3 = nn.Linear(int(input_size/4),3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hybrid = Hybrid(qiskit.Aer.get_backend('aer_simulator'), 100, np.pi / 2)

    def forward(self, x):
        output = self.l1(x) 
        output = self.relu(output)
        output = self.l2(output)
        output = self.l3(output)
        output = self.hybrid(output)
        output = output / 100
        return output


        
        
def life_step(X, organisms, i, times, population_cnt):
    """
    Function which describes one step in the animation. This
    function takes care of counting populations, allowing
    organisms to die and reproduce, and modeling the organisms
    on an image to be shown during the animation. 
    """
    times.append(i)
    quantum_pop_cnt = np.sum([organism.quantum for organism in organisms])
    classical_pop_cnt = len(organisms) - quantum_pop_cnt
    population_cnt["quantum"].append(quantum_pop_cnt)
    population_cnt["classical"].append(classical_pop_cnt)
    #population_cnt.append(len(organisms))
    
    if(np.random.rand()<PROBABILITY_OF_NEW_CLUMP):
        size = len(X)
        rx = np.random.randint(0, size-CLUMP_SIZE)
        ry = np.random.randint(0,size-CLUMP_SIZE)
        X[rx:rx+CLUMP_SIZE, ry:ry+CLUMP_SIZE] = np.ones((CLUMP_SIZE, CLUMP_SIZE)) * 4
        
    
    added_organisms = []
    removed_organisms = []
    Y = np.array([[X[i][j] if X[i][j]>3 else 0 for j in range(len(X))] for i in range(len(X))])
    for organism in organisms:
        if organism.time_until_death < 0:
            removed_organisms.append(organism)
            #print("died :(")
            continue
        
        organism.move(X)
        
        if (Y[organism.pos[0], organism.pos[1]] == 4):
            organism.food += 1
            organism.time_until_death = FOOD_REGENERATION
        
        if (organism.food >= AMOUNT_OF_FOOD_TO_REPRODUCE):
            for i in range(NUMBER_OF_CHILDREN):
                added_organisms.append(organism.reproduce(X))
            organism.food = 0
            
        organism.time_until_death -= 1
        
        if organism.quantum:
            Y[organism.pos[0], organism.pos[1]] = 2.75
        else:
            Y[organism.pos[0], organism.pos[1]] = 1.25
    X = Y
    
    for organism in added_organisms:
        organisms.append(organism)
    for organism in removed_organisms:
        organisms.remove(organism)
        
    return X

def animate(X, organisms, life_step, frames, times, population_cnt, cmap='gist_stern'):

  fig = plt.figure()
  img = plt.imshow(X, vmin=0, cmap=cmap)
  plt.close()

  def update(i):
    global X
    
    if i % 10 == 0:
        print(i/ANIMATION_LENGTH*100, '%')

    X = life_step(X, organisms, i, times, population_cnt)
    img.set_array(X)

    return img

  return FuncAnimation(fig, update, frames=frames, interval=50).to_jshtml()



#Excecution code: creates the world, fills it with initial prey
#                 and predators, and then runs the animal

X = np.zeros((size, size))


for i in range(INITIAL_NUMBER_OF_FOOD_CLUMPS):
    rx = np.random.randint(0, size-CLUMP_SIZE)
    ry = np.random.randint(0,size-CLUMP_SIZE)
    X[rx:rx+CLUMP_SIZE, ry:ry+CLUMP_SIZE] = np.ones((CLUMP_SIZE, CLUMP_SIZE)) * 4

organisms = []
for i in range(NUM_CLASSICAL_PREDATORS):
    organisms.append(Predator(0, 
                         [np.random.randint(0, size), np.random.randint(0,size)], 
                         directions[np.random.randint(0, 4)]))
for i in range(NUM_QUANTUM_PREDATORS):
    organisms.append(Predator(0, 
                         [np.random.randint(0, size), np.random.randint(0,size)], 
                         directions[np.random.randint(0, 4)], quantum=True))

times = []
population_cnt = {"quantum":[], "classical":[]}
animation = animate(X, organisms, life_step, ANIMATION_LENGTH, times, population_cnt)
plt.plot(times, population_cnt["quantum"], label="Quantum Population", color="orange")
plt.plot(times, population_cnt["classical"], label="Classical Population", color="purple")
plt.legend()
plt.savefig("output/population_plot" + str(SEED) + ".png")
#plt.show()  #uncomment if running interactively


with open('output/animation' + str(SEED) + '.html', 'w') as f:
    f.write(animation)
#IPython.display.HTML(animation) #uncomment if running interactively