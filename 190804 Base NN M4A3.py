import torch
import torchvision
import torchvision.transforms as transforms
import torch.testing
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch.optim as optim
import copy
import time
from pandas import DataFrame
from datetime import datetime
import winsound

import torch.nn as nn
import torch.nn.functional as F

# setting the width of the pictures
torch.set_printoptions(linewidth=120)
# turned on by default, but nevertheless
torch.set_grad_enabled(True)

# getting the data in
# Extract, Transform, Load!
train_set = torchvision.datasets.MNIST(
    # specifying where the data is
    root='./data/MINST'
    # it is for training data
    , train=True
    # download if it is not in the location specified by the root
    , download=True
    # transforming the data into a tensor
    , transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# getting test set
test_set = torchvision.datasets.MNIST(
    # specifying where the data is
    root='./data/FashionMINST'
    # it is for training data
    , train=False
    # download if it is not in the location specified by the root
    , download=True
    # transforming the data into a tensor
    , transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


# a function need to for adding more Y axis to a graphs (see 190803 Multiple Y axis folder)
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


# a function checking how correct the prediction is
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# a function to define moving average
def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# shorthand way of flattening the tensor
def flatten(t):
    t = t.reshape(1,-1)
    # the -1 tell the pytorch to figure out what the value should be
    t= t.squeeze()
    return t

# creating the network


# it utilises nn.Module to connect to PyTorch usability, making it a subclass of Module class
class Network(nn.Module):
    def __init__(self):
        # call to the super class Module from nn
        super(Network, self).__init__()

        # fc strand for 'fully connected'
        self.fc1 = nn.Linear(in_features=28*28, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.out = nn.Linear(in_features=100, out_features=10)

    def forward(self, t):

        # (1) input layer (redundant)
        t = t

        # (2) hidden linear layer
        # As my t consists of 28*28 bit pictures, I need to flatten them:
        t = t.reshape(-1, 28*28)
        # Now having this reshaped input, add it to the linear layer
        t = self.fc1(t)
        # Again, apply ReLU as the activation function
        t = F.relu(t)  # F.leaky_relu F.relu(t)

        # (3) hidden linear layer
        # As above, but reshaping is not needed now
        t = self.fc2(t)
        t = F.relu(t)  # F.leaky_relu

        # (4) output layer
        t = self.out(t)
        # softmax is not needed here, but could be in different function
#        t = F.softmax(t, dim=1)

        return t


# define batch size for the training
outer_b_size = 128
inner_b_size = 128
# define size of the training data (set to len(train_set) for all data to be used)
train_size = len(train_set)
# define size of the testing data (set to len(test_set) for all data to be used)
test_size = len(test_set)
# define how many epochs should be done
outer_epochs = 1
inner_epochs = 1


# length o moving average
len_mov_avg = 5
# define the learning rate
outer_learning_rate = 0.1
inner_learning_rate = 0.1
# how many full test/data checks we want to do, evenly spaced on log scale (the actual number is bigger by one)
full_checks = 10
# counter for controling which step we are having
counter = 1
# a check for not calculating outer G weights difference on the first go
check_w_diff = False
# storage for the previous G end weights from the previous inner SGD run
np_weights_G_end_prev = []
# storage for the previous weight at the standard path of outer SGD, to check what is the expected step size for end-end difference
np_weight_loss_start_prev = []
# Defining if we work on GPU or CPU
GPU = False
# defining if we print a sample batch
show_batch = False
# defining if we print a training data
show_train = False
# defining if we print a test data
show_test = False
# defining if we want to have deterministic results, no variability
determinism = True
# defining if we want to export the results to CSV
exporting_to_csv = False
# defining if we want to keep memory heavy data, i.e. weights and gradients for all batches
keep_heavy = False
# Defining if we want to run Saddle Checks
SC = True
# Defining if we want to print ongoing G evolutions, whenever checks are performed
G_evol_print = False
# array for recording batch loss
a_batch_loss = []
# array for recording number of correct hits in a batch
a_batch_num_correct = []
# array for recording whole train data loss
a_train_loss = []
# array for recording whole test data loss
a_test_loss = []
# array for recording whole train data loss, with zeros when no check was made
a_train_loss_long = []
# array for recording whole test data loss, with zeros when no check was made
a_test_loss_long = []
# array for recording whole train data number of correct hits
a_train_num_correct = []
# array for recording whole test data number of correct hits
a_test_num_correct = []
# array for recording weights
a_weights = []
# array for recording gradient of the loss, on the error minimisation path
a_G_start = []
# array for recording gradient of the loss, after minimisation and trying to find the saddle point
a_G_end = []
# array for recording the loss on the full training data at the ending, minimum G
a_G_end_loss = []
# array for storing the difference of weights at the start of G to the end G, within one inner SGD
a_weights_G_diff_inter = []
# array for storing the difference in G end weights of consecutive inner SGD
a_weights_G_diff_outer = []
# array for storing the ending G weights only
a_weights_G_end = []
# array for storing the differences between the goes of the outer SGD
a_weights_loss_start_diff = []
# array for storing the weights which were used for all training data
a_train_weights = []
# array for storing the gradients which were obtained when testing on all training data
a_train_G = []
# array for recording the sum of the absolute loss gradients, on all training data
a_train_abs_sum_G = []
# array of arrays, which will be recording evolutions of G batches
a_a_G_evol = []

# implement the effects of determinism
if determinism is True:
    seed_choosen = 0

    np.random.seed(seed_choosen)
    torch.manual_seed(seed_choosen)

# create points at which full train/test network run should be taken, as well as define points for the saddle check
A = np.arange(0, full_checks+1)
X_full_check = np.round(np.power(10, np.log10(np.floor((train_size/outer_b_size)*outer_epochs))*(A/full_checks))) # np.arange(100, 469+1)
X_full_check = np.unique(X_full_check)
# Defining saddle check  range, where inner runs will be performed to find the saddles
SC_start = 10
SC_end = 13
X_SC_outer = np.arange(SC_start, SC_end+1)

# decide on which device to use
device = torch.device("cuda:0" if GPU is True else "cpu")
# record starting time
start_time = time.time()

# **************** DATA LOADERS **********************
print('Creating data loaders and full train/test batches...')

# create a subsets of the whole data set to be actually used for training (could be all data)
train_subset = torch.utils.data.Subset(train_set, np.arange(0, train_size))

# creating a data loader to get a batches for training, shuffling their order
outer_batch_train_loader = torch.utils.data.DataLoader(
    train_subset
    , batch_size=outer_b_size
    , shuffle=True
)

inner_batch_train_loader = torch.utils.data.DataLoader(
    train_subset
    , batch_size=inner_b_size
    , shuffle=False
)

# creating a data loader to get a the whole train data
full_train_loader = torch.utils.data.DataLoader(
    train_subset
    , batch_size=len(train_subset)
)
# get the whole train data
full_train_batch = next(iter(full_train_loader))
# Splitting the train data
full_train_images, full_train_labels = full_train_batch[0].to(device), full_train_batch[1].to(device)

# creating a data loader to get a the whole test data
full_test_loader = torch.utils.data.DataLoader(
    test_set
    , batch_size=test_size
)
# get the whole test data
full_test_batch = next(iter(full_test_loader))
# Splitting the test data
full_test_images, full_test_labels = full_test_batch[0].to(device), full_test_batch[1].to(device)

# **************** MAIN PART **********************
# build the network, assign to the correct device
outer_model = Network()
outer_model.to(device)
# Create optimizer for optimising gradients
outer_optimizer = optim.SGD(outer_model.parameters(), lr=outer_learning_rate)
# size of the data
print('training set size:', train_size)
# loop for the number of epochs
for b in range(outer_epochs):
    print('***** EPOCH NO. ', b+1)
    # getting a batch iterator
    outer_batch_iterator = iter(outer_batch_train_loader)
    # For loop for a single epoch, based on the length of the training set and the batch size
    for a in range(round(train_size/outer_b_size)):
        print(a+1)
        # zeroing previous gradient
        outer_optimizer.zero_grad()
        # get one batch for the iteration
        outer_batch = next(outer_batch_iterator)
        # decomposing a batch
        images, labels = outer_batch[0].to(device), outer_batch[1].to(device)
        # to get a prediction, as with individual layers, we need to equate it to the network with the samples as input:
        preds = outer_model(images)
        # with the predictions, we will use F to get the loss as cross_entropy
        loss = F.cross_entropy(preds, labels)
        # calculate the gradients needed for update of weights, by back propagation
        loss.backward()
        # with the known weights, step in the direction of correct estimation
        outer_optimizer.step()
        # Now, test how the network does now, using the new weights NOT PRINTED NOW
        preds = outer_model(images)
        loss = F.cross_entropy(preds, labels)
        # record batch loss, turning the tensor into a number before recording
        a_batch_loss.append(loss.item())
        # record correct predictions
        a_batch_num_correct.append(get_num_correct(preds, labels))
        # record advanced results
        # create tensor to which all weights will be added
        t_weights = torch.tensor([]) # dtype=torch.float64
        t_gradients = torch.tensor([])
        for param in outer_model.parameters():
            # prepare results as a flat tensor, i.e. vector
            weights_to_stack = flatten(param.data)
            gradients_to_stack = flatten(param.grad)
            #print(param.grad)
            #print(param.grad.size())
            # merge the previous layer results with the new layer
            t_weights = torch.cat([t_weights, weights_to_stack], dim=0)
            t_gradients = torch.cat([t_gradients, gradients_to_stack], dim=0)
            #print(param.data)
        # Record weights and all other elements for advanced analysis
        csv_weights = t_weights.numpy() # network.fc2.weight.data.numpy()
        csv_G = t_gradients.numpy()
        new_csv_weights = copy.deepcopy(csv_weights)
        new_csv_G = copy.deepcopy(csv_G)
        # append the lists accordingly, depending if we want to keep heavy data
        if keep_heavy is True:
            a_weights.append(new_csv_weights)
            a_G_start.append(new_csv_G)
        a_train_loss_long.append('')
        a_test_loss_long.append('')
        # check if whole train/test data check should be performed
        if counter in X_full_check:
            # get the result on the whole train data and record them
            full_train_preds = outer_model(full_train_images)
            full_train_loss = F.cross_entropy(full_train_preds, full_train_labels)
            a_train_loss.append(full_train_loss.item())
            # Get a proportion of correct estimates, to make them comparable between train and test data
            full_train_num_correct = get_num_correct(full_train_preds, full_train_labels) / train_size
            a_train_num_correct.append(full_train_num_correct)
            print('Correct predictions of the dataset:', full_train_num_correct)
            # get the results for the whole test data
            full_test_preds = outer_model(full_test_images)
            full_test_loss = F.cross_entropy(full_test_preds, full_test_labels)
            a_test_loss.append(full_test_loss.item())
            full_test_num_correct = get_num_correct(full_test_preds, full_test_labels) / test_size
            a_test_num_correct.append(full_test_num_correct)
            # For long results, remove the last item and add a correct number
            del a_train_loss_long[-1]
            del a_test_loss_long[-1]
            a_train_loss_long.append(full_train_loss.item())
            a_test_loss_long.append(full_test_loss.item())
            print(counter)
            # dod the saddle check if required
        if SC is True and counter in X_SC_outer:
            # ********************** START INNER LOOP ***********************
            print(' *** SADDLE SEARCH ***')
            print('Batch #:', counter)
            # create a deep copy of the current outer model, for the inner SGD loop
            inner_model = copy.deepcopy(outer_model)
            # create optimiser to work on the inner model
            inner_optimizer = optim.SGD(inner_model.parameters(), lr=inner_learning_rate)
            # recording the starting G on all training data:
            # the predictions for the whole training data
            inner_full_train_preds = inner_model(full_train_images)
            # calculating the standard loss function, as I will have to know what is the loss at this point
            inner_full_train_loss = F.cross_entropy(inner_full_train_preds,
                                         full_train_labels)
            # calculating gradients IN A NEW WAY, which allows working with different functions
            inner_full_train_loss_grads = torch.autograd.grad(inner_full_train_loss, inner_model.parameters(),
                                                   create_graph=True)
            # compute the squared norm of the loss gradient, which is my G
            G = 0.5 * sum([grd.norm() ** 2 for grd in inner_full_train_loss_grads])
            # Record the starting G, based on the full training set
            print('starting G: ', G.item())
            a_G_start.append(G.item())
            # recording the starting G weights
            t_weights_G_start = torch.tensor([])  # dtype=torch.float64
            for param in inner_model.parameters():
                # prepare results as a flat tensor, i.e. vector
                weights_to_stack = flatten(param.data)
                # print(param.data)
                # merge the previous layer results with the new layer
                t_weights_G_start = torch.cat([t_weights_G_start, weights_to_stack], dim=0)
                # print(param.data)
            # Record weights
            np_weights_G_start = t_weights_G_start.numpy()
            print('Weights at starting G:', np_weights_G_start)
            print(len(np_weights_G_start))
            # new_csv_weights = copy.deepcopy(csv_weights)
            # create an array which will be recording the evolution of G over one inner SGD
            a_G_evol = []
            # go to the main inner SGD
            for b_inner in range(inner_epochs):
                # getting a batch iterator, for the inner loop
                inner_batch_iterator = iter(inner_batch_train_loader)
                for a_inner in range(round(train_size/inner_b_size)):
                    # zeroing previous gradient
                    inner_optimizer.zero_grad()
                    # getting a batch for training
                    inner_batch = next(inner_batch_iterator)
                    # decomposing a batch
                    inner_batch_images, inner_batch_labels = inner_batch[0].to(device), inner_batch[1].to(device)
                    # the result for the whole training data
                    inner_batch_preds = inner_model(inner_batch_images)
                    # calculating the standard loss function, as I will have to know what is the loss at this point
                    inner_batch_loss = F.cross_entropy(inner_batch_preds,
                                             inner_batch_labels)
                    # calculating gradients IN A NEW WAY, which allows working with different functions
                    inner_batch_loss_grads = torch.autograd.grad(inner_batch_loss, inner_model.parameters(),
                                                       create_graph=True)
                    # compute the squared norm of the loss gradient, which is my G
                    G = 0.5 * sum([grd.norm() ** 2 for grd in inner_batch_loss_grads])
                    # Record the starting G, based on the full training set
                    #print(G)
                    # add this starting G to the evolution array
                    a_G_evol.append(G.item())
                    # calculate second gradient, or gradient of G, or GG, but it is not used directly (can change "G" to "loss" to have the standard algorythm)
                    G.backward()
                    # make a step in the direction of the gradients
                    inner_optimizer.step()
            # record the final, saddle point G, for the whole data!
            inner_full_train_preds = inner_model(full_train_images)
            # calculating the standard loss function, as I will have to know what is the loss at this point
            inner_full_train_loss = F.cross_entropy(inner_full_train_preds,
                                                    full_train_labels)
            # calculating gradients IN A NEW WAY, which allows working with different functions
            inner_full_train_loss_grads = torch.autograd.grad(inner_full_train_loss, inner_model.parameters(),
                                                              create_graph=True)
            # compute the squared norm of the loss gradient, which is my G
            G = 0.5 * sum([grd.norm() ** 2 for grd in inner_full_train_loss_grads])
            # Record the starting G, based on the full training set
            print('Ending G: ', G.item())
            a_G_end.append(G.item())
            # recording the weights, both for the ending G weights and also for the starting loss, from the outer model
            t_weights_G_end = torch.tensor([])
            t_weights_loss_start = torch.tensor([])
            for param in inner_model.parameters():
                # prepare results as a flat tensor, i.e. vector
                weights_to_stack = flatten(param.data)
                # print(param.data)
                # merge the previous layer results with the new layer
                t_weights_G_end = torch.cat([t_weights_G_end, weights_to_stack], dim=0)
                # print(param.data)
            for param in outer_model.parameters():
                # prepare results as a flat tensor, i.e. vector
                weights_to_stack = flatten(param.data)
                # print(param.data)
                # merge the previous layer results with the new layer
                t_weights_loss_start = torch.cat([t_weights_loss_start, weights_to_stack], dim=0)
                # print(param.data)
            # Record weights
            np_weights_G_end = t_weights_G_end.numpy()
            np_weights_loss_start = t_weights_loss_start.numpy()
            print('Weights at starting loss:', np_weights_loss_start)
            #print('Weights at ending G:', np_weights_G_end)
            # if not the first run, get the difference, get norm and record it, otherwise mark that the first run was done
            if check_w_diff is True:
                # get the difference vector
                np_weights_G_diff_outer = np_weights_G_end - np_weights_G_end_prev
                np_weights_loss_start_diff = np_weights_loss_start - np_weight_loss_start_prev
                # calculate the norm of the difference vector
                weights_G_diff_outer = np.linalg.norm(np_weights_G_diff_outer)
                weights_loss_start_diff = np.linalg.norm(np_weights_loss_start_diff)
                print('The weight difference between this step of outer SGD and the previous one:', weights_loss_start_diff)
                a_weights_G_diff_outer.append(weights_G_diff_outer)
                a_weights_loss_start_diff.append(weights_loss_start_diff)

            else:
                print('No previous result to compare to!')
                check_w_diff = True
            # Record the end G weights for the next run update
            np_weights_G_end_prev = np_weights_G_end
            np_weight_loss_start_prev = np_weights_loss_start
            # For the inner weight_G difference, get the difference vector and the take its norm to calculate the distance between the two solutions
            np_weights_G_diff_inter = np_weights_G_end-np_weights_G_start
            #print('The difference between starting and ending weights:', np_weights_G_diff_inter)
            # get the actual distance between the begining and an end, by using the norm of the vector
            weights_G_diff_inter = np.linalg.norm(np_weights_G_diff_inter)
            #print('The distance between the begining and ending weights:', weights_G_diff_inter)
            # Record the difference in W
            a_weights_G_diff_inter.append(weights_G_diff_inter)
            # Record the evolution of G
            a_a_G_evol.append(a_G_evol)
            # when requested, print the evolution of G, for this particular batch
            if G_evol_print is True:
                X_SC_inner_length = len(a_G_evol)
                X_SC_inner = np.linspace(0, X_SC_inner_length, X_SC_inner_length)
                fig, ax1 = plt.subplots()
                plot_G_evol = ax1.plot(X_SC_inner, a_G_evol, 'red', label='G evolution')
                plt.title('G evolution at outer batch no. ')
                plt.legend()
                plt.show()
            # obtain the full training loss at that final point
            preds_G_end = inner_model(full_train_images)
            loss_G_end = F.cross_entropy(preds_G_end, full_train_labels)
            a_G_end_loss.append(loss_G_end)
            print(' *** SADDLE SEARCH END ***')
        counter = counter + 1

# make a signal that the simulation is done
frequency = 800  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
# Printing experiment information
print('********** EXPERIMENT INFORMATION **********')
print('Network:', outer_model)
print('Number of test images:', len(full_test_images))
print('Number of train images:', len(full_train_images))
# print learning rate
print('Outer learning rate:', outer_learning_rate)
# print epoch number
print('Epochs:', outer_epochs)
# print seed
print('seed:', seed_choosen)

# Results for the test dataset
print('***** TRAINING RESULTS *****')
full_train_preds = outer_model(full_train_images)
# checking how many correct labels were given on the test batch
#print('Correct labels of training data:', full_train_labels)
# showing the current predictions on test batch
#print('Predictions of training data after update:', full_train_preds.argmax(dim=1))
# the accuracy of the network
print('Proportion of correct training classification after update:', get_num_correct(full_train_preds, full_train_labels)/len(full_train_preds))

# Results for the test dataset
print('***** TEST RESULTS *****')
full_test_preds = outer_model(full_test_images)
# checking how many correct labels were given on the test batch
#print('Correct labels of test data:', full_test_labels)
# showing the current predictions on test batch
#print('Predictions of test data after update:', full_test_preds.argmax(dim=1))
# the accuracy of the network
print('Correct test classification after update:', get_num_correct(full_test_preds, full_test_labels)/len(full_test_preds))
# print simulation time
print('Simulation time:'"--- %s seconds ---" % (time.time() - start_time))
# print what device was used
print('Device used:', device)
# export the data to CSV if requested
if exporting_to_csv is True:
    to_export = {'X_full_check': X_full_check,
            'a_train_loss': a_train_loss,
            'a_test_loss': a_test_loss,
            'a_train_num_correct': a_train_num_correct,
            'a_test_num_correct': a_test_num_correct,
            'a_train_weights': a_train_weights,
            'a_train_G': a_train_G,
            'a_train_abs_sum_G': a_train_abs_sum_G
            }
    # check if heavy dataset should be exported
    if keep_heavy is True:
        to_export_advanced = {'a_weights': a_weights,
                         'a_G_start': a_G_start,
                         'a_G_start': a_G_start,
                         'a_train_loss_long': a_train_loss_long,
                         'a_test_loss_long': a_test_loss_long
                         }
    else:
        to_export_advanced = {'a_G_start': a_G_start,
                              'a_train_loss_long': a_train_loss_long,
                              'a_test_loss_long': a_test_loss_long
                              }
    df_adv = DataFrame(to_export_advanced)
    df = DataFrame(to_export)# columns= ['X_full_check', 'a_train_num_correct']
    now = datetime.now()
    time_label = now.strftime('%y%m%d_%H-%M-%S')
    file_name_base = time_label + 'Base_results' + '.csv'
    file_name_weights = time_label + 'Advanced_results' + '.csv'
    df.to_csv(file_name_base, index=None)
    df_adv.to_csv(file_name_weights, index=None)
    print('Results have been exported to .csv')

# Plotting loss and accuracy,
X_SC_weights_G_diff_outer = X_SC_outer[1:]
X_SC_avg = moving_average(X_SC_outer, len_mov_avg)

fig, ax1 = plt.subplots()
plot_train_loss = ax1.plot(X_full_check, a_train_loss, 'darkgreen', label='Train loss')
plot_test_loss = ax1.plot(X_full_check, a_test_loss, 'yellowgreen', label='Test loss')
plt.legend()
ax2 = ax1.twinx()
plot_train_accuracy = ax2.plot(X_full_check, a_train_num_correct, 'royalblue', label='Train accuracy')
plot_test_accuracy = ax2.plot(X_full_check, a_test_num_correct, 'deepskyblue', label='Test accuracy')
plt.title('Loss and accuracy evolution')
plt.legend()
plt.show()

if SC is True:
    # Plot the results of error and G
    fig1, ax1 = plt.subplots()
    plot_train_loss, = ax1.plot(X_full_check, a_train_loss, 'darkgreen', label='Train loss')
    plot_test_loss, = ax1.plot(X_full_check, a_test_loss, 'yellowgreen', label='Test loss')
    plot_G_loss, = ax1.plot(X_SC_outer, a_G_end_loss, 'yellow', label = 'G end loss')
    plt.legend()
    ax2 = ax1.twinx()
    plot_G_start, = ax2.plot(X_SC_outer, a_G_start, 'red', label='G start')
    plot_G_start_avg, = ax2.plot(X_SC_avg, moving_average(a_G_start, len_mov_avg), 'darkred', label='G start average')
    plot_G_end, = ax2.plot(X_SC_outer, a_G_end, 'salmon', label='G end')
    plot_G_end_avg, = ax2.plot(X_SC_avg, moving_average(a_G_end, len_mov_avg), 'lightsalmon', label='G end average')
    plt.legend()
    # adding additional Y axis and shifting it for it to make sense (see 190803 multiple Y axis folder)
    ax3 = ax1.twinx()
    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    ax3.spines["right"].set_position(("axes", 1.05))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(ax3)
    # Second, show the right spine.
    ax3.spines["right"].set_visible(True)
    plot_G_diff_weight_inner, = ax3.plot(X_SC_outer, a_weights_G_diff_inter, 'm', label='Start-End diff in weights')
    plot_G_diff_weight_outer, = ax3.plot(X_SC_weights_G_diff_outer, a_weights_G_diff_outer, 'violet', label='End-End G diff in weights')
    plot_loss_diff_weight, = ax3.plot(X_SC_weights_G_diff_outer, a_weights_loss_start_diff, 'blue', label='out SGD loss diff in weights')
    # The difference between the movements of the outer SGD and the end Gs
    a_weights_SP_dispersion = [a_weights_G_diff_outer[i]-a_weights_loss_start_diff[i] for i in range(min(len(a_weights_G_diff_outer), len(a_weights_loss_start_diff)))]
    plot_weight_SP_dispersion, = ax3.plot(X_SC_weights_G_diff_outer, a_weights_SP_dispersion, 'indigo',
                                      label='Saddle dispersion from outer SGD') # positive = separate from the previous saddle, negative = attractive saddle, zero = wide saddle
    print('Weights dispersion SC:', a_weights_SP_dispersion)
    # non-essential graphics boosters
    plt.title('Loss, G and weights evolution')
    plt.legend()

    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("G")
    ax3.set_ylabel("Weights difference")

    ax1.yaxis.label.set_color(plot_train_loss.get_color())
    ax2.yaxis.label.set_color(plot_G_start.get_color())
    ax3.yaxis.label.set_color(plot_G_diff_weight_inner.get_color())

    tkw = dict(size=4, width=1.5)
    ax1.tick_params(axis='y', colors=plot_train_loss.get_color(), **tkw)
    ax2.tick_params(axis='y', colors=plot_G_start.get_color(), **tkw)
    ax3.tick_params(axis='y', colors=plot_G_diff_weight_inner.get_color(), **tkw)
    plt.show()

    # Plot individual batch G evolutions, REMOVED FOR NOW AS CAN'T CHANGE AXIS EFFECTIVELY
    # X_SC_inner_length = len(a_a_G_evol[0])
    # X_SC_inner = np.linspace(0, X_SC_inner_length, X_SC_inner_length)
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    #
    # # defining lines to be plotted
    # # z_line = np.linspace(0.6, 0, 1000)  # z-axis = value of loss or G
    # # x_line = np.linspace(0, 1000, 1000)  # x-axis = outer SGD
    # # y_line = np.linspace(0, 0, 1000)  # y-axis = inner SGD
    # # ax.plot3D(x_line, y_line, z_line, 'green')
    #
    # # now all these new lines are to be printed on the graph
    # # ax = fig.add_subplot(projection='3d')
    # for a in range(len(a_a_G_evol)):
    #     z_line_2 = a_a_G_evol[a]
    #     x_line_2 = np.linspace(X_full_check[a], X_full_check[a], X_SC_inner_length)
    #     y_line_2 = np.linspace(0, X_SC_inner_length, X_SC_inner_length)
    #     ax.plot3D(x_line_2, y_line_2, z_line_2, 'red')
    #
    # plt.show()

# show data requested
if show_batch is True:
    # show an example the batch
    batch_iterator = iter(outer_batch_train_loader)
    batch = next(batch_iterator)
    # decomposing a batch
    images_batch, labels_batch = batch[0], batch[1]
    grid = torchvision.utils.make_grid(images_batch, nrow=10)
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()
if show_train is True:
    # removing the tensor from GPU for printing
    full_train_images = full_train_images.to('cpu')
    grid = torchvision.utils.make_grid(full_train_images, nrow=10)
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()
if show_test is True:
    # removing the tensor from GPU for printing
    full_test_images = full_test_images.to('cpu')
    grid = torchvision.utils.make_grid(full_test_images, nrow=10)
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()
