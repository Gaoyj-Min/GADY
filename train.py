import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from modules.gan import Generator
from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder, get_data_settings
from utils.data_processing2 import get_data, compute_time_statistics
torch.autograd.set_detect_anomaly(True)

### Argument and global variables
parser = argparse.ArgumentParser('PINT self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='uci')
parser.add_argument('--seed', type=int, default=142, help='seed')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_epoch', type=int, default=1, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=1, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=100, help='Dimensions of the memory for '
                                                                'each user') # was 172, 100 for uci
parser.add_argument('--use_gan', type=bool, default = False, help=  "whether to use gan") # was 172, 100 for uci
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features'),
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')

parser.add_argument('--beta', type=float, default=0.1,
                    help='Initial value for the beta parameter in PINT')
parser.add_argument('--r_dim', type=int, default=4, help='dim of positional features')
parser.add_argument('--anomaly_per', type=float, default=0.01, help='the anomaly rate in the test data')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)
use_gan = args.use_gan



### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, test_data = get_data(args.data,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features, anomaly_per = args.anomaly_per)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
# val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
# nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
#                                       seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
# nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
#                                        new_node_test_data.destinations,
#                                        seed=3)
# Set device
device_string = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
performance_best_auc = 0
performance_best_ap = 0
train_generator = Generator(batch_size = args.bs)

for i in range(args.n_runs):
  results_path = f"results/pint-{args.data}_{i}.pkl" if args.prefix == '' else f"results/{args.prefix}_{i}.pkl"
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=args.n_layer, use_memory=args.use_memory,
            message_dimension=args.message_dim, memory_dimension=args.memory_dim,
            memory_update_at_start=not args.memory_update_at_end,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=args.n_degree,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            beta=args.beta,
            r_dim=args.r_dim)
  criterion = torch.nn.BCELoss()
  d_optimizer = torch.optim.Adam(tgn.parameters(), lr=args.lr)
  g_optimizer = torch.optim.Adam(tgn.parameters(), lr=args.lr)
  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / args.bs)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  test_aps = []
  test_aucs = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []
  generator_losses = []
  early_stopper = EarlyStopMonitor(max_round=args.patience)

  next_V, next_R = [], []

  test_V, test_R = [], []

  for epoch in range(args.n_epoch):
    tgn.reset_VR()
    start_epoch = time.time()

    ### Training
    # Reinitialize memory of the model at the start of each epoch
    if args.use_memory:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []
    m_loss2 = []

    logger.info('start {} epoch'.format(epoch))
    for k in range(0, num_batch, args.backprop_every):
      d_loss = 0
      g_loss = 0
      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * args.bs
        end_idx = min(num_instance, start_idx + args.bs)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        edges = np.hstack((sources_batch.reshape(-1,1), destinations_batch.reshape(-1,1)))
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]
        batch_data = np.hstack((edges,timestamps_batch.reshape(-1,1)))
        size = len(sources_batch)
        sources_batch = torch.tensor(sources_batch)
        destinations_batch = torch.tensor(destinations_batch)
        timestamps_batch = torch.tensor(timestamps_batch)

        with torch.no_grad():
          pos_label = torch.zeros(size, dtype=torch.float, device=device)
          neg_label = torch.ones(size, dtype=torch.float, device=device)
 
        tgn = tgn.train()

        partition_size, _ = get_data_settings(args.data)

        idx = k
        if (k % partition_size == 0) and args.data: # savepoint
          prt = k // partition_size
          next_V, next_R = torch.load( 'pos_features/' + args.data +'_nextVR_part_' + str(prt) + '_bs_' + str(args.bs) +
                                       '_rdim_' +str(args.r_dim) + str(args.anomaly_per))
          for c in range(len(next_V)):
            next_V[c] = next_V[c].to(device)
            next_R[c] = next_R[c].to(device)
        if use_gan:
          g_optimizer.zero_grad()
          neg_samples = train_generator(batch_data)
          neg_prob = tgn.compute_edge_probabilities(neg_samples[:,0], neg_samples[:,1], neg_samples[:,2],
                                    edge_idxs_batch, args.n_degree,update_memory=False,
                                    next_V=tgn.V + next_V[idx%partition_size].to_dense(),
                                    next_R=tgn.R + next_R[idx%partition_size].to_dense())
          g_loss = criterion(neg_prob.squeeze(), pos_label)
          g_loss.backward()
          g_optimizer.step()
          
          # sources_batch_neg, destinations_batch_neg, edge_time_neg = np.array(neg_samples[:,0]), np.array(neg_samples[:,1]), np.array(neg_samples[:,2])
          # sources_batch_neg, destinations_batch_neg, edge_time_neg = neg_samples[:,0], neg_samples[:,1], neg_samples[:,2]
          # pos_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, args.n_degree,
          # d_real_loss.backward()
          # x = train_generator(batch_data).detach()  
          # head = torch.floor(x[:,0]*(max(batch_data[:,0])-min(batch_data[:,0])) + min(batch_data[:,0]))
          # tail = torch.floor(x[:,1]*(max(batch_data[:,1])-min(batch_data[:,1])) + min(batch_data[:,1]))
          # time = torch.floor(x[:,2]*(max(batch_data[:,2])-min(batch_data[:,2])) + min(batch_data[:,2]))
          # data = torch.hstack([head.reshape(-1,1),tail.reshape(-1,1),time.reshape(-1,1)]).detach().numpy()
          # data = data[np.lexsort([data.T[-1]])] 
          # data = data.astype(int)
          # sources_batch_neg, destinations_batch_neg, edge_time_neg = data[:,0], data[:,1], data[:,2]
          # sources_batch_neg, destinations_batch_neg, edge_time_neg = train_generator(batch_data).detach() 
 
          d_optimizer.zero_grad()
          pos_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, args.n_degree,
                                          update_memory=True,
                                    next_V=tgn.V + next_V[idx%partition_size].to_dense(),
                                    next_R=tgn.R + next_R[idx%partition_size].to_dense()) 
          neg_prob2 = tgn.compute_edge_probabilities(neg_samples[:,0].detach(), neg_samples[:,1].detach(),
                                              neg_samples[:,2].detach(), edge_idxs_batch, args.n_degree,update_memory=False,
                                              next_V=tgn.V + next_V[idx%partition_size].to_dense(),
                                              next_R=tgn.R + next_R[idx%partition_size].to_dense())
          
          # neg_prob = tgn.compute_edge_probabilities(sources_batch_neg, destinations_batch_neg,edge_time_neg,

          d_fake_loss = criterion(neg_prob2.squeeze(), neg_label)
          d_real_loss = criterion(pos_prob.squeeze(), pos_label)
          # d_fake_loss.backward()
          d_loss = (d_fake_loss + d_real_loss)/2
          d_loss.backward()
          d_optimizer.step()
          
          
          # neg_samples2 = train_generator(batch_data)
          # sources_batch_neg2, destinations_batch_neg2, edge_time_neg2 = np.array(neg_samples2[:,0]), np.array(neg_samples2[:,1]), np.array(neg_samples2[:,2])
          # neg_prob = tgn.compute_edge_probabilities(sources_batch_neg2, destinations_batch_neg2,
          #                                     edge_time_neg2, edge_idxs_batch, args.n_degree,update_memory=False,
          #                                     next_V=tgn.V + next_V[idx%partition_size].to_dense(),
          #                                     next_R=tgn.R + next_R[idx%partition_size].to_dense())

          
        else:
          sources_batch_neg, destinations_batch_neg = train_rand_sampler.sample(edges)
          pos_prob = tgn.compute_edge_probabilities2(sources_batch, destinations_batch, timestamps_batch,edge_time_neg, edge_idxs_batch, args.n_degree,
                                    next_V=tgn.V + next_V[idx%partition_size].to_dense(),
                                    next_R=tgn.R + next_R[idx%partition_size].to_dense())
          d_real_loss = criterion(pos_prob.squeeze(), pos_label) 
          neg_prob = tgn.compute_edge_probabilities(sources_batch_neg, destinations_batch_neg,
                                              edge_time_neg, edge_idxs_batch, args.n_degree,
                                              next_V=tgn.V + next_V[idx%partition_size].to_dense(),
                                              next_R=tgn.R + next_R[idx%partition_size].to_dense())
          d_fake_loss = criterion(neg_prob.squeeze(), neg_label)
          d_loss = d_fake_loss + d_real_loss
          # pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, sources_batch_neg, destinations_batch_neg,
          #                                             timestamps_batch, edge_idxs_batch, args.n_degree,
          #                                             next_V=tgn.V + next_V[idx%partition_size].to_dense(),
          #                                             next_R=tgn.R + next_R[idx%partition_size].to_dense())
        
        
        # loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

      d_loss /= args.backprop_every
      g_loss /= args.backprop_every
      # if use_gan:
      #   d_real_loss.backward()
      #   d_fake_loss.backward()
      #   d_optimizer.step()
      #   g_loss.backward()
      #   g_optimizer.step()
      # else:
      #   d_loss.backward()
      #   d_optimizer.step()
      m_loss.append(d_loss.item())
      m_loss2.append(g_loss.item())

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      if args.use_memory:
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    train_V, train_R = tgn.V, tgn.R  # save state at end of training
    
    
    test_V, test_R = torch.load('pos_features/' + args.data + '_VR_test_bs_' + str(args.bs) +
                                                    '_rdim_' +str(args.r_dim) +str(args.anomaly_per))
    ### Validation
    # Validation uses the full graph
    tgn.set_neighbor_finder(full_ngh_finder)

    if args.use_memory:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      train_memory_backup = tgn.memory.backup_memory()

    test_ap, test_auc, _, _ = eval_edge_prediction(model=tgn, data=test_data,
                                                 n_neighbors=args.n_degree, vs=test_V, rs=test_R)
    if args.use_memory:
      val_memory_backup = tgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test edges are
      # strictly later in time than validation edges)
      tgn.memory.restore_memory(train_memory_backup)

    # new_nodes_val_aps.append(nn_val_ap)
    test_aucs.append(test_auc)
    test_aps.append(test_ap)
    train_losses.append(np.mean(m_loss))
    generator_losses.append(np.mean(m_loss2))
    if test_ap > performance_best_ap:
      performance_best_ap = test_ap
    if test_auc > performance_best_auc:
      performance_best_auc = test_auc
    # Save temporary results to disk
    pickle.dump({
      "test_aps": test_aps,
      "test_aucs": test_aucs,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('Epoch mean G_loss: {}'.format(np.mean(m_loss2)))
    logger.info(
      'test auc: {}'.format(test_auc))
    logger.info(
      'test ap: {}'.format(test_ap))
    logger.info(
      'best auc: {}'.format(performance_best_auc))
    logger.info(
      'best ap: {}'.format(performance_best_ap))
    # Early stopping
    if early_stopper.early_stop_check(test_ap):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      tgn.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      break
    else:
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

  test_R, test_V, next_V, next_R = [], [], [], []  # cleaning the memory

  if args.use_memory:
    val_memory_backup = tgn.memory.backup_memory()
    # Restore memory we had at the end of training to be used when validating on new nodes.
    # Also backup memory after validation so it can be used for testing (since test edges are
    # strictly later in time than validation edges)
    tgn.memory.restore_memory(train_memory_backup) 


  ### Test
  test_V, test_R = torch.load('pos_features/' + args.data + '_VR_test_bs_' + str(args.bs) +
                                                      '_rdim_' +str(args.r_dim) +str(args.anomaly_per))
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  tgn.V, tgn.R = train_V, train_R
  test2_ap, test2_auc, _, _ = eval_edge_prediction(model=tgn,
                                                 data=test_data, n_neighbors=args.n_degree,
                                                 vs=test_V, rs=test_R)

  logger.info(
    'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))

  # Save results for this run
  pickle.dump({
    "test_aps": test_aps,
    "test_aucs": test_aucs,
    "test2_ap": test2_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  logger.info('Saving PINT model')
  if args.use_memory:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(train_memory_backup)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  logger.info('PINT model saved')
